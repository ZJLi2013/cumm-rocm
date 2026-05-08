"""Kernel-only profiler driver for implicit GEMM K-pipe variants.

This script intentionally bypasses the high-level forward loop after one compile
warmup. The profiled region repeatedly launches the cached FlyDSL kernel with
prebuilt mask/LUT/packed weights so profiler output focuses on the implicit GEMM
kernel, not Python reference code, mask generation, or output allocation.
"""
import argparse
import os
import sys
import time

import torch


def make_subm_pairs(n_active, kv, density, device):
    nhot_per_kv = []
    max_nhot = 0
    for k in range(kv):
        if k == kv // 2:
            nhot = n_active
        else:
            nhot = int(n_active * density)
        nhot_per_kv.append(nhot)
        max_nhot = max(max_nhot, nhot)

    indice_pairs = -torch.ones(kv, 2, max_nhot, dtype=torch.int32, device=device)
    indice_pair_num = torch.zeros(kv, dtype=torch.int32, device=device)

    for k, nhot in enumerate(nhot_per_kv):
        indice_pair_num[k] = nhot
        inp = torch.randperm(n_active, device=device, dtype=torch.int32)[:nhot]
        out = torch.randperm(n_active, device=device, dtype=torch.int32)[:nhot]
        if k == kv // 2:
            out = inp.clone()
        indice_pairs[k, 0, :nhot] = inp
        indice_pairs[k, 1, :nhot] = out

    return indice_pairs, indice_pair_num


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["direct", "remap"], default=os.getenv("KPIPE_VARIANT", "direct"))
    parser.add_argument("--n-active", type=int, default=int(os.getenv("N_ACTIVE", "20000")))
    parser.add_argument("--c-in", type=int, default=int(os.getenv("C_IN", "64")))
    parser.add_argument("--c-out", type=int, default=int(os.getenv("C_OUT", "128")))
    parser.add_argument("--kv", type=int, default=int(os.getenv("KV", "27")))
    parser.add_argument("--density", type=float, default=float(os.getenv("DENSITY", "0.3")))
    parser.add_argument("--warmup", type=int, default=int(os.getenv("WARMUP", "10")))
    parser.add_argument("--iters", type=int, default=int(os.getenv("ITERS", "200")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "0")))
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from cumm.implicit_gemm_common import _dtype_to_str, _get_hip_module, _pack_weights
    from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe import (
        MFMA_F32_16X16X4F32_N2_ASHARED_KPIPE_COMPILED_KERNELS,
        implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward,
        implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_remap_forward,
    )

    torch.manual_seed(args.seed)
    device = "cuda"
    dtype = torch.float32
    dtype_str = _dtype_to_str(dtype)
    block_m = 16
    block_k = 16
    epilogue = "remap" if args.variant == "remap" else "direct"
    name = "mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16"
    if epilogue != "direct":
        name = f"{name}_{epilogue}"

    features = torch.randn(args.n_active, args.c_in, dtype=dtype, device=device) * 0.1
    filters = torch.randn(args.kv, args.c_in, args.c_out, dtype=dtype, device=device) * 0.1
    indice_pairs, indice_pair_num = make_subm_pairs(
        args.n_active, args.kv, args.density, device
    )

    forward = (
        implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_remap_forward
        if args.variant == "remap"
        else implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward
    )

    # Compile and validate that the target variant is available before profiling.
    compiled_out = forward(features, filters, indice_pairs, indice_pair_num, args.n_active)
    if compiled_out is None:
        raise RuntimeError(f"{name} failed to compile")
    torch.cuda.synchronize()

    key = (name, args.c_in, args.c_out, args.kv, dtype_str, block_k, epilogue)
    launch_fn, block_m = MFMA_F32_16X16X4F32_N2_ASHARED_KPIPE_COMPILED_KERNELS[key]

    hip = _get_hip_module()
    if hip is None:
        raise RuntimeError("HIP extension is not available")
    _, _, _, mask, _, _, inp_row_lut = hip.build_implicit_gemm_mask(
        indice_pairs, indice_pair_num, args.n_active, block_m
    )
    num_tiles = (args.n_active + block_m - 1) // block_m
    features_c = features.contiguous()
    weights_packed = _pack_weights(filters, 16)
    lut_flat = inp_row_lut.reshape(-1).contiguous()
    mask_flat = mask.reshape(-1).contiguous()
    out_features = torch.empty(args.n_active, args.c_out, dtype=torch.float32, device=device)

    def run_kernel():
        launch_fn(
            features_c,
            weights_packed,
            out_features,
            lut_flat,
            mask_flat,
            num_tiles,
            args.n_active,
            torch.cuda.current_stream(),
        )

    for _ in range(args.warmup):
        run_kernel()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.iters):
        run_kernel()
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) * 1e6 / max(args.iters, 1)

    print(
        "profile_kpipe_kernel",
        f"variant={args.variant}",
        f"shape=N{args.n_active}_C{args.c_in}_{args.c_out}_KV{args.kv}_D{args.density}",
        f"iters={args.iters}",
        f"avg_us={elapsed_us:.3f}",
        f"out_shape={tuple(out_features.shape)}",
    )


if __name__ == "__main__":
    main()
