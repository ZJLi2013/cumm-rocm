"""FlyDSL implicit GEMM kernel family public entry."""

from dataclasses import dataclass
from typing import Callable, List, Optional

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
from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2 import (
    MFMA_F32_16X16X4F32_N2_COMPILED_KERNELS,
    _compile_implicit_gemm_mfma_f32_16x16x4f32_n2,
    implicit_gemm_mfma_f32_16x16x4f32_n2_forward,
)
from cumm.implicit_gemm_mfma_f32_32x32x2f32 import (
    MFMA_F32_32X32X2F32_COMPILED_KERNELS,
    _compile_implicit_gemm_mfma_f32_32x32x2f32,
    implicit_gemm_mfma_f32_32x32x2f32_forward,
)
from cumm.implicit_gemm_scalar_tile import (
    SCALAR_TILE_COMPILED_KERNELS,
    _compile_implicit_gemm_scalar_tile,
    implicit_gemm_scalar_tile_forward,
)


@dataclass(frozen=True)
class ImplicitGemmKernelDesp:
    """Runtime-visible descriptor for one implicit GEMM family member."""

    name: str
    forward: Callable
    dtype: str
    min_c_in_multiple: int
    min_c_out_multiple: int
    block_m: int
    block_n: int
    mfma_shape: Optional[str]
    priority: int

    def is_available(self, dtype_str: str, c_in: int, c_out: int) -> bool:
        if dtype_str != self.dtype and self.dtype != "any":
            return False
        if c_in % self.min_c_in_multiple != 0:
            return False
        return c_out % self.min_c_out_multiple == 0


IMPLICIT_GEMM_KERNELS = {
    "scalar_tile": implicit_gemm_scalar_tile_forward,
    "mfma_f32_16x16x4f32": implicit_gemm_mfma_f32_16x16x4f32_forward,
    "mfma_f32_16x16x4f32_n2": implicit_gemm_mfma_f32_16x16x4f32_n2_forward,
    "mfma_f32_32x32x2f32": implicit_gemm_mfma_f32_32x32x2f32_forward,
}


IMPLICIT_GEMM_KERNEL_DESPS: List[ImplicitGemmKernelDesp] = [
    ImplicitGemmKernelDesp(
        name="mfma_f32_16x16x4f32_n2",
        forward=implicit_gemm_mfma_f32_16x16x4f32_n2_forward,
        dtype="f32",
        min_c_in_multiple=4,
        min_c_out_multiple=32,
        block_m=16,
        block_n=32,
        mfma_shape="16x16x4f32",
        priority=100,
    ),
    ImplicitGemmKernelDesp(
        name="mfma_f32_16x16x4f32",
        forward=implicit_gemm_mfma_f32_16x16x4f32_forward,
        dtype="f32",
        min_c_in_multiple=4,
        min_c_out_multiple=16,
        block_m=16,
        block_n=16,
        mfma_shape="16x16x4f32",
        priority=90,
    ),
    ImplicitGemmKernelDesp(
        name="mfma_f32_32x32x2f32",
        forward=implicit_gemm_mfma_f32_32x32x2f32_forward,
        dtype="f32",
        min_c_in_multiple=2,
        min_c_out_multiple=32,
        block_m=32,
        block_n=32,
        mfma_shape="32x32x2f32",
        priority=80,
    ),
    ImplicitGemmKernelDesp(
        name="scalar_tile",
        forward=implicit_gemm_scalar_tile_forward,
        dtype="any",
        min_c_in_multiple=1,
        min_c_out_multiple=1,
        block_m=64,
        block_n=C_OUT_TILE_MAX,
        mfma_shape=None,
        priority=0,
    ),
]


def _dtype_to_dispatch_str(dtype) -> str:
    if str(dtype).endswith("float32"):
        return "f32"
    if str(dtype).endswith("float16"):
        return "f16"
    if str(dtype).endswith("bfloat16"):
        return "bf16"
    return str(dtype)


def get_implicit_gemm_candidates(dtype, c_in: int, c_out: int) -> List[ImplicitGemmKernelDesp]:
    """Return kernel candidates in dispatch order for a shape."""
    dtype_str = _dtype_to_dispatch_str(dtype)
    candidates = [
        desp
        for desp in IMPLICIT_GEMM_KERNEL_DESPS
        if desp.is_available(dtype_str, c_in, c_out)
    ]
    return sorted(candidates, key=lambda desp: desp.priority, reverse=True)


def select_implicit_gemm_kernel(dtype, c_in: int, c_out: int) -> ImplicitGemmKernelDesp:
    """Select the preferred kernel descriptor before compile/runtime fallback."""
    return get_implicit_gemm_candidates(dtype, c_in, c_out)[0]


def implicit_gemm_forward(features, filters, indice_pairs, indice_pair_num, num_activate_out):
    """Dispatch implicit GEMM across the current kernel family.

    The dispatch policy is intentionally conservative: use MFMA candidates when
    shape constraints match, but fall through to scalar_tile if compile/runtime
    availability rejects a candidate.
    """
    c_in = features.shape[1]
    c_out = filters.shape[2]
    for desp in get_implicit_gemm_candidates(features.dtype, c_in, c_out):
        out = desp.forward(features, filters, indice_pairs, indice_pair_num, num_activate_out)
        if out is not None:
            return out
    return None
