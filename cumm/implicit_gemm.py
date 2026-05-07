"""FlyDSL implicit GEMM kernel family public entry."""

from cumm.implicit_gemm_common import (
    C_OUT_TILE_MAX,
    _get_hip_module,
    _pack_weights,
    preprocess_pairs,
)
from cumm.implicit_gemm_mfma_f32_16x16x4f32 import (
    MFMA_F32_16X16X4F32_COMPILED_KERNELS,
    _compile_implicit_gemm_mfma_f32_16x16x4f32,
    implicit_gemm_mfma_f32_16x16x4f32_forward,
)
from cumm.implicit_gemm_scalar_tile import (
    SCALAR_TILE_COMPILED_KERNELS,
    _compile_implicit_gemm_scalar_tile,
    implicit_gemm_scalar_tile_forward,
)


def implicit_gemm_forward(*args, **kwargs):
    """Default implicit GEMM entry.

    Keep the conservative scalar family as the default until MFMA dispatch
    policy is implemented and benchmarked.
    """
    return implicit_gemm_scalar_tile_forward(*args, **kwargs)


IMPLICIT_GEMM_KERNELS = {
    "scalar_tile": implicit_gemm_scalar_tile_forward,
    "mfma_f32_16x16x4f32": implicit_gemm_mfma_f32_16x16x4f32_forward,
}
