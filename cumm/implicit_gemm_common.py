"""Shared utilities for FlyDSL implicit GEMM kernels."""
import math
import os
import warnings
from typing import Optional, Dict, Tuple

import torch

_HIP_MODULE = None


def _get_hip_module():
    """JIT compile the HIP indice pairs + mask generation extension."""
    global _HIP_MODULE
    if _HIP_MODULE is not None:
        return _HIP_MODULE

    try:
        from torch.utils.cpp_extension import load
        csrc_dir = os.path.join(os.path.dirname(__file__), 'csrc_hip')
        _HIP_MODULE = load(
            name='cumm_hip_indice',
            sources=[
                os.path.join(csrc_dir, 'indice_pairs_api.cpp'),
                os.path.join(csrc_dir, 'indice_pairs_kernel.hip'),
            ],
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3'],
            verbose=False,
        )
        return _HIP_MODULE
    except Exception as e:
        warnings.warn(f"Failed to compile HIP indice pairs module: {e}")
        return None


def _dtype_to_str(dtype: torch.dtype) -> Optional[str]:
    if dtype == torch.float32:
        return 'f32'
    elif dtype == torch.float16:
        return 'f16'
    elif dtype == torch.bfloat16:
        return 'bf16'
    return None


def _ensure_flydsl_path():
    """Add FlyDSL to sys.path if needed."""
    import sys
    _flydsl_root = os.path.dirname(os.path.dirname(__import__('flydsl').__file__))
    if _flydsl_root not in sys.path:
        sys.path.insert(0, _flydsl_root)
    if '/opt/FlyDSL' not in sys.path and os.path.isdir('/opt/FlyDSL'):
        sys.path.insert(0, '/opt/FlyDSL')


def _forward_common(
    features: torch.Tensor,
    filters: torch.Tensor,
    indice_pairs: torch.Tensor,
    indice_pair_num: torch.Tensor,
    num_activate_out: int,
    version_tag: str,
    compiled_kernels: Dict,
    compile_fn,
    needs_sorted_arrays: bool = False,
):
    """Common forward logic for V3-V6+.

    Returns output tensor or None on failure.
    """
    try:
        import flydsl
    except ImportError:
        return None

    hip = _get_hip_module()
    if hip is None:
        return None

    kv = filters.shape[0]
    c_in = features.shape[1]
    c_out = filters.shape[2]
    device = features.device
    dtype = features.dtype

    if c_in % 4 != 0 or c_out % 4 != 0:
        return None

    dtype_str = _dtype_to_str(dtype)
    if dtype_str is None:
        return None

    BLOCK_M = 64
    num_tiles = (num_activate_out + BLOCK_M - 1) // BLOCK_M

    sorted_inp, sorted_out, sorted_kv, mask, pair_start, pair_end, inp_row_lut = \
        hip.build_implicit_gemm_mask(indice_pairs, indice_pair_num,
                                     num_activate_out, BLOCK_M)

    if sorted_inp.shape[0] == 0:
        return torch.zeros(num_activate_out, c_out, dtype=torch.float32, device=device)

    key = (version_tag, c_in, c_out, kv, dtype_str)
    if key not in compiled_kernels:
        try:
            launch_fn, block_m = compile_fn(c_in, c_out, kv, dtype_str)
            compiled_kernels[key] = (launch_fn, block_m)
        except Exception as e:
            import traceback
            warnings.warn(f"Failed to compile {version_tag} implicit GEMM kernel: {e}\n{traceback.format_exc()}")
            return None
    else:
        launch_fn, block_m = compiled_kernels[key]

    out_features = torch.zeros(num_activate_out, c_out, dtype=torch.float32, device=device)
    weights_flat = filters.reshape(-1).contiguous()
    stream = torch.cuda.current_stream()

    if needs_sorted_arrays:
        launch_fn(
            features.contiguous(), weights_flat, out_features,
            sorted_inp, sorted_out,
            mask.reshape(-1).contiguous(),
            pair_start.reshape(-1).contiguous(),
            pair_end.reshape(-1).contiguous(),
            num_tiles, num_activate_out, stream,
        )
    else:
        launch_fn(
            features.contiguous(), weights_flat, out_features,
            inp_row_lut.reshape(-1).contiguous(),
            mask.reshape(-1).contiguous(),
            num_tiles, num_activate_out, stream,
        )

    if dtype != torch.float32:
        out_features = out_features.to(dtype)

    return out_features


def preprocess_pairs(
    indice_pairs: torch.Tensor,
    indice_pair_num: torch.Tensor,
    block_m: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Legacy preprocessing (kv-centric). Kept for testing."""
    kv = indice_pairs.shape[0]
    device = indice_pairs.device
    pair_num_cpu = indice_pair_num.cpu().int().numpy()

    total_pairs = int(pair_num_cpu.sum())
    if total_pairs == 0:
        empty = torch.zeros(0, dtype=torch.int32, device=device)
        return empty, empty, empty, empty, 0

    num_tiles_per_kv = [(int(n) + block_m - 1) // block_m if n > 0 else 0
                        for n in pair_num_cpu]
    num_tiles = sum(num_tiles_per_kv)
    if num_tiles == 0:
        empty = torch.zeros(0, dtype=torch.int32, device=device)
        return empty, empty, empty, empty, 0

    all_inp = torch.empty(total_pairs, dtype=torch.int32, device=device)
    all_out = torch.empty(total_pairs, dtype=torch.int32, device=device)
    all_kv = torch.empty(total_pairs, dtype=torch.int32, device=device)

    offset = 0
    for k in range(kv):
        nhot = int(pair_num_cpu[k])
        if nhot <= 0:
            continue
        all_inp[offset:offset + nhot] = indice_pairs[k, 0, :nhot]
        all_out[offset:offset + nhot] = indice_pairs[k, 1, :nhot]
        all_kv[offset:offset + nhot] = k
        offset += nhot

    max_idx = int(all_inp.max().item()) + 1
    sort_key = all_kv.long() * max_idx + all_inp.long()
    sorted_order = torch.argsort(sort_key)
    all_inp = all_inp[sorted_order]
    all_out = all_out[sorted_order]

    total_padded = num_tiles * block_m
    inp_flat = torch.zeros(total_padded, dtype=torch.int32, device=device)
    out_flat = torch.zeros(total_padded, dtype=torch.int32, device=device)
    tile_kpos = torch.empty(num_tiles, dtype=torch.int32, device=device)
    tile_pair_count = torch.empty(num_tiles, dtype=torch.int32, device=device)

    tile_idx = 0
    flat_offset = 0
    src_offset = 0
    for k in range(kv):
        nhot = int(pair_num_cpu[k])
        if nhot <= 0:
            continue
        inp_flat[flat_offset:flat_offset + nhot] = all_inp[src_offset:src_offset + nhot]
        out_flat[flat_offset:flat_offset + nhot] = all_out[src_offset:src_offset + nhot]
        n_tiles_k = num_tiles_per_kv[k]
        tile_kpos[tile_idx:tile_idx + n_tiles_k] = k
        for t in range(n_tiles_k):
            n_valid = min(block_m, nhot - t * block_m)
            tile_pair_count[tile_idx + t] = n_valid
        tile_idx += n_tiles_k
        flat_offset += n_tiles_k * block_m
        src_offset += nhot

    return inp_flat, out_flat, tile_kpos, tile_pair_count, num_tiles
