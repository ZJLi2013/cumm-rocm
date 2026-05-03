"""FlyDSL-based GEMM Tuner — drop-in replacement for cumm's GemmTunerSimple.

This module provides the same interface that spconv expects from cumm's GEMM layer,
but routes all computation through FlyDSL's hgemm_splitk kernel on AMD GPUs.

Usage:
    from cumm.gemm_tuner import FlyDSLGemmTuner
    tuner = FlyDSLGemmTuner()
    tuner.matmul(c, a, b)  # C = A @ B^T
"""

from typing import Optional, Dict, Tuple
import torch

# FlyDSL kernel import (requires flydsl installed)
from kernels.hgemm_splitk import hgemm_splitk_, get_default_kwargs

# gfx942 constraints:
#   WARP_ATOM_M = WARP_ATOM_N = 16, BLOCK_M_WARPS = BLOCK_N_WARPS = 2
#   => TILE_M >= 32, TILE_N >= 32, and n % TILE_N == 0
#   LDS (B_TO_LDS=True): STAGES * (TILE_M + TILE_N) * TILE_K * 2 <= 65536
#   Also: max(above, TILE_M * TILE_N * 2) <= 65536
_SMEM_CAPACITY_GFX942 = 65536
_STAGES = 2
_DTYPE_BYTES = 2
_WARP_ATOM = 16
_BLOCK_WARPS = 2
_MIN_TILE = _BLOCK_WARPS * _WARP_ATOM  # 32


def _smem_use(tile_m: int, tile_n: int, tile_k: int) -> int:
    """Compute LDS usage (bytes) matching FlyDSL's hgemm_splitk formula."""
    ab = _STAGES * (tile_m * tile_k + tile_n * tile_k) * _DTYPE_BYTES
    return max(ab, tile_m * tile_n * _DTYPE_BYTES)


def _select_tile_config(m: int, n: int, k: int) -> Dict[str, int]:
    """Select tile configuration for given problem shape.

    Uses FlyDSL's get_default_kwargs as baseline, then adjusts TILE_N for
    divisibility and ensures LDS fits gfx942 budget.
    """
    defaults = get_default_kwargs(m, n, k)
    tile_m = defaults['TILE_M']
    tile_n = defaults['TILE_N']
    tile_k = defaults['TILE_K']
    split_k = defaults['SPLIT_K']

    # Ensure TILE_N divides n (FlyDSL requires n % BLOCK_N == 0)
    if n % tile_n != 0:
        for candidate in [128, 64, _MIN_TILE]:
            if n % candidate == 0:
                tile_n = candidate
                break
        else:
            tile_n = _MIN_TILE

    # Ensure TILE_M >= 32 (warp constraint)
    tile_m = max(tile_m, _MIN_TILE)

    # Shrink to fit gfx942 LDS (65536 bytes)
    while _smem_use(tile_m, tile_n, tile_k) > _SMEM_CAPACITY_GFX942:
        if tile_m > _MIN_TILE:
            tile_m = tile_m // 2
        elif tile_n > _MIN_TILE:
            tile_n = tile_n // 2
        elif tile_k > 32:
            tile_k = tile_k // 2
        else:
            break

    # k must be divisible by TILE_K * SPLIT_K for splitk correctness
    while split_k > 1 and k % (tile_k * split_k) != 0:
        split_k -= 1

    return {
        'TILE_M': tile_m,
        'TILE_N': tile_n,
        'TILE_K': tile_k,
        'SPLIT_K': split_k,
        'BLOCK_M_WARPS': _BLOCK_WARPS,
        'BLOCK_N_WARPS': _BLOCK_WARPS,
    }


class FlyDSLGemmTuner:
    """Drop-in replacement for cumm GemmTunerSimple.

    Provides the GEMM interface expected by spconv's Native path:
      gather → GEMM → scatter

    The gather/scatter is handled by spconv externally (S2a approach).
    This tuner only needs to do standard dense matmul: C = A @ B^T.
    """

    def __init__(self):
        self._cache: Dict[Tuple[int, int, int, str], Dict] = {}

    def matmul(
        self,
        c: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        beta: float = 0.0,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        """Compute C = alpha * A @ B^T + beta * C.

        Args:
            c: Output tensor [M, N], fp16 or bf16
            a: Input tensor [M, K], fp16 or bf16
            b: Weight tensor [N, K], fp16 or bf16 (note: N-major, transposed internally)
            beta: Accumulation factor (0.0 = overwrite, 1.0 = accumulate)
            stream: CUDA stream (optional)
        """
        m, k = a.shape
        n = b.shape[0]
        assert b.shape[1] == k, f"Shape mismatch: a=[{m},{k}], b=[{n},{b.shape[1]}]"
        assert c.shape == (m, n), f"Output shape mismatch: expected ({m},{n}), got {c.shape}"

        if beta != 0.0:
            # FlyDSL hgemm_splitk overwrites C. For beta=1.0 accumulation,
            # we compute into a temp buffer and add.
            tmp = torch.zeros_like(c)
            kwargs = self._get_kwargs(m, n, k, a.dtype)
            hgemm_splitk_(tmp, a, b, hgemm_kwargs=kwargs)
            c.add_(tmp)
        else:
            kwargs = self._get_kwargs(m, n, k, a.dtype)
            hgemm_splitk_(c, a, b, hgemm_kwargs=kwargs)

    def _get_kwargs(self, m: int, n: int, k: int, dtype: torch.dtype) -> Dict:
        """Get or compute tile configuration (cached)."""
        dtype_key = "f16" if dtype == torch.float16 else "bf16"
        cache_key = (m, n, k, dtype_key)
        if cache_key not in self._cache:
            self._cache[cache_key] = _select_tile_config(m, n, k)
        return self._cache[cache_key]
