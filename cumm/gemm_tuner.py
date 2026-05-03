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

# Valid TILE_N values for hgemm_splitk
_VALID_TILE_N = [64, 128, 256]
# Valid TILE_K values (64 is safest for LDS on gfx942)
_VALID_TILE_K = [64, 128, 256]


def _select_tile_n(n: int) -> int:
    """Select largest TILE_N that evenly divides N."""
    for tn in reversed(_VALID_TILE_N):
        if n % tn == 0:
            return tn
    raise ValueError(
        f"N={n} is not divisible by any valid TILE_N {_VALID_TILE_N}. "
        f"spconv channels should be multiples of 64."
    )


def _select_tile_config(m: int, n: int, k: int) -> Dict[str, int]:
    """Select tile configuration for given problem shape.

    Heuristic tuning for spconv typical shapes (tall-skinny: large M, small N/K).
    """
    tile_n = _select_tile_n(n)
    tile_k = 64  # safe default, fits gfx942 LDS budget

    # Tile M selection based on problem size
    if m <= 32:
        tile_m = 16
    elif m <= 128:
        tile_m = 32
    elif m <= 512:
        tile_m = 64
    else:
        tile_m = 128

    # Split-K: useful when M is small relative to K (backward weight)
    split_k = 1
    if m <= 64 and k >= 1024:
        split_k = min(16, k // tile_k)

    return {
        'TILE_M': tile_m,
        'TILE_N': tile_n,
        'TILE_K': tile_k,
        'SPLIT_K': split_k,
        'BLOCK_M_WARPS': 2,
        'BLOCK_N_WARPS': 2,
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
