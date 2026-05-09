"""FlyDSL implicit GEMM kernel family public entry."""

import os
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
from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2_ashared import (
    MFMA_F32_16X16X4F32_N2_ASHARED_COMPILED_KERNELS,
    _compile_implicit_gemm_mfma_f32_16x16x4f32_n2_ashared,
    implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_forward,
)
from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe import (
    MFMA_F32_16X16X4F32_N2_ASHARED_KPIPE_COMPILED_KERNELS,
    _compile_implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe,
    implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward,
    implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_remap_forward,
    implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk32_forward,
    implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_forward,
)
from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_db import (
    MFMA_F32_16X16X4F32_N2_ASHARED_KPIPE_DB_COMPILED_KERNELS,
    _compile_implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_db,
    implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_db32_forward,
    implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_db64_forward,
)
from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_crossk import (
    CROSSK_COMPILED_KERNELS,
    implicit_gemm_crossk_forward,
    implicit_gemm_crossk_prefetch_forward,
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
    tile_m: int
    tile_n: int
    block_k: Optional[int]
    waves: int
    ashared: bool
    epilogue: str
    mfma_shape: Optional[str]
    priority: int

    def is_available(self, dtype_str: str, c_in: int, c_out: int) -> bool:
        if dtype_str != self.dtype and self.dtype != "any":
            return False
        if c_in % self.min_c_in_multiple != 0:
            return False
        return c_out % self.min_c_out_multiple == 0

    @property
    def block_m(self) -> int:
        """Backward-compatible alias while callers migrate to tile_m."""
        return self.tile_m

    @property
    def block_n(self) -> int:
        """Backward-compatible alias while callers migrate to tile_n."""
        return self.tile_n


def _crossk_pf_xor_bk32_forward(features, filters, indice_pairs, indice_pair_num, num_activate_out):
    c_in = features.shape[1]
    c_out = filters.shape[2]
    kv = filters.shape[0]
    if c_in < 32 or c_in % 32 != 0:
        return None
    if kv < 15:
        return None
    if c_in * c_out < 2048:
        return None
    # Large channels: crossk preprocessing overhead exceeds launch-overhead savings.
    # Fall back to native (at::mm) until per-KV implicit GEMM is implemented.
    if c_in * c_out > 8192:
        return None
    return implicit_gemm_crossk_prefetch_forward(
        features, filters, indice_pairs, indice_pair_num, num_activate_out,
        block_k=32, use_xor_swizzle=True,
    )


IMPLICIT_GEMM_KERNELS = {
    "scalar_tile": implicit_gemm_scalar_tile_forward,
    "mfma_f32_16x16x4f32": implicit_gemm_mfma_f32_16x16x4f32_forward,
    "mfma_f32_16x16x4f32_n2": implicit_gemm_mfma_f32_16x16x4f32_n2_forward,
    "mfma_f32_16x16x4f32_n2_ashared": implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_forward,
    "mfma_f32_16x16x4f32_n2_ashared_kpipe": (
        implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_forward
    ),
    "mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16": (
        implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward
    ),
    "mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_remap": (
        implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_remap_forward
    ),
    "mfma_f32_16x16x4f32_n2_ashared_kpipe_bk32": (
        implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk32_forward
    ),
    "mfma_f32_16x16x4f32_n2_ashared_kpipe_db32": (
        implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_db32_forward
    ),
    "mfma_f32_16x16x4f32_n2_ashared_kpipe_db64": (
        implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_db64_forward
    ),
    "mfma_f32_32x32x2f32": implicit_gemm_mfma_f32_32x32x2f32_forward,
    "crossk_bk32": implicit_gemm_crossk_forward,
    "crossk_pf_xor_bk32": _crossk_pf_xor_bk32_forward,
}


IMPLICIT_GEMM_KERNEL_DESPS: List[ImplicitGemmKernelDesp] = [
    ImplicitGemmKernelDesp(
        name="crossk_pf_xor_bk32",
        forward=_crossk_pf_xor_bk32_forward,
        dtype="f32",
        min_c_in_multiple=32,
        min_c_out_multiple=32,
        tile_m=16,
        tile_n=32,
        block_k=32,
        waves=2,
        ashared=True,
        epilogue="direct",
        mfma_shape="16x16x4f32",
        priority=98,
    ),
    ImplicitGemmKernelDesp(
        name="mfma_f32_16x16x4f32_n2",
        forward=implicit_gemm_mfma_f32_16x16x4f32_n2_forward,
        dtype="f32",
        min_c_in_multiple=4,
        min_c_out_multiple=32,
        tile_m=16,
        tile_n=32,
        block_k=None,
        waves=2,
        ashared=False,
        epilogue="direct",
        mfma_shape="16x16x4f32",
        priority=100,
    ),
    ImplicitGemmKernelDesp(
        name="mfma_f32_16x16x4f32_n2_ashared",
        forward=implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_forward,
        dtype="f32",
        min_c_in_multiple=4,
        min_c_out_multiple=32,
        tile_m=16,
        tile_n=32,
        block_k=None,
        waves=2,
        ashared=True,
        epilogue="direct",
        mfma_shape="16x16x4f32",
        priority=95,
    ),
    ImplicitGemmKernelDesp(
        name="mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16",
        forward=implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward,
        dtype="f32",
        min_c_in_multiple=4,
        min_c_out_multiple=32,
        tile_m=16,
        tile_n=32,
        block_k=16,
        waves=2,
        ashared=True,
        epilogue="direct",
        mfma_shape="16x16x4f32",
        priority=94,
    ),
    ImplicitGemmKernelDesp(
        name="mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_remap",
        forward=implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_remap_forward,
        dtype="f32",
        min_c_in_multiple=4,
        min_c_out_multiple=32,
        tile_m=16,
        tile_n=32,
        block_k=16,
        waves=2,
        ashared=True,
        epilogue="remap",
        mfma_shape="16x16x4f32",
        priority=92,
    ),
    ImplicitGemmKernelDesp(
        name="mfma_f32_16x16x4f32_n2_ashared_kpipe_bk32",
        forward=implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk32_forward,
        dtype="f32",
        min_c_in_multiple=4,
        min_c_out_multiple=32,
        tile_m=16,
        tile_n=32,
        block_k=32,
        waves=2,
        ashared=True,
        epilogue="direct",
        mfma_shape="16x16x4f32",
        priority=93,
    ),
    ImplicitGemmKernelDesp(
        name="mfma_f32_16x16x4f32_n2_ashared_kpipe",
        forward=implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_forward,
        dtype="f32",
        min_c_in_multiple=4,
        min_c_out_multiple=32,
        tile_m=16,
        tile_n=32,
        block_k=32,
        waves=2,
        ashared=True,
        epilogue="direct",
        mfma_shape="16x16x4f32",
        priority=78,
    ),
    ImplicitGemmKernelDesp(
        name="mfma_f32_16x16x4f32_n2_ashared_kpipe_db32",
        forward=implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_db32_forward,
        dtype="f32",
        min_c_in_multiple=4,
        min_c_out_multiple=32,
        tile_m=16,
        tile_n=32,
        block_k=32,
        waves=2,
        ashared=True,
        epilogue="direct",
        mfma_shape="16x16x4f32",
        priority=80,
    ),
    ImplicitGemmKernelDesp(
        name="mfma_f32_16x16x4f32_n2_ashared_kpipe_db64",
        forward=implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_db64_forward,
        dtype="f32",
        min_c_in_multiple=4,
        min_c_out_multiple=32,
        tile_m=16,
        tile_n=32,
        block_k=64,
        waves=2,
        ashared=True,
        epilogue="direct",
        mfma_shape="16x16x4f32",
        priority=79,
    ),
    ImplicitGemmKernelDesp(
        name="mfma_f32_16x16x4f32",
        forward=implicit_gemm_mfma_f32_16x16x4f32_forward,
        dtype="f32",
        min_c_in_multiple=4,
        min_c_out_multiple=16,
        tile_m=16,
        tile_n=16,
        block_k=None,
        waves=1,
        ashared=False,
        epilogue="direct",
        mfma_shape="16x16x4f32",
        priority=90,
    ),
    ImplicitGemmKernelDesp(
        name="mfma_f32_32x32x2f32",
        forward=implicit_gemm_mfma_f32_32x32x2f32_forward,
        dtype="f32",
        min_c_in_multiple=2,
        min_c_out_multiple=32,
        tile_m=32,
        tile_n=32,
        block_k=None,
        waves=1,
        ashared=False,
        epilogue="direct",
        mfma_shape="32x32x2f32",
        priority=80,
    ),
    ImplicitGemmKernelDesp(
        name="scalar_tile",
        forward=implicit_gemm_scalar_tile_forward,
        dtype="any",
        min_c_in_multiple=1,
        min_c_out_multiple=1,
        tile_m=64,
        tile_n=C_OUT_TILE_MAX,
        block_k=None,
        waves=1,
        ashared=False,
        epilogue="direct",
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
    by_name = {desp.name: desp for desp in candidates}
    forced_kernel = os.environ.get("CUMM_IMPLICIT_GEMM_KERNEL", "auto").strip()
    if forced_kernel and forced_kernel != "auto":
        if forced_kernel not in IMPLICIT_GEMM_KERNELS:
            raise ValueError(f"Unknown implicit GEMM kernel: {forced_kernel}")
        if forced_kernel not in by_name:
            raise ValueError(
                f"Implicit GEMM kernel {forced_kernel} is not available for "
                f"dtype={dtype_str}, C_in={c_in}, C_out={c_out}"
            )
        return [by_name[forced_kernel]]

    if dtype_str == "f32" and c_in <= 16:
        preferred_names = [
            "mfma_f32_16x16x4f32_n2_ashared",
            "mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16",
            "mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_remap",
            "mfma_f32_16x16x4f32",
            "mfma_f32_16x16x4f32_n2",
            "scalar_tile",
        ]
    elif dtype_str == "f32" and c_out % 32 == 0:
        preferred_names = [
            "crossk_pf_xor_bk32",
            "mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16",
            "scalar_tile",
            "mfma_f32_16x16x4f32_n2_ashared",
            "mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_remap",
            "mfma_f32_16x16x4f32_n2_ashared_kpipe_bk32",
            "mfma_f32_16x16x4f32",
            "mfma_f32_16x16x4f32_n2",
        ]
    else:
        preferred_names = [
            "scalar_tile",
            "mfma_f32_16x16x4f32_n2",
            "mfma_f32_16x16x4f32_n2_ashared",
            "mfma_f32_16x16x4f32",
        ]

    ordered = [by_name[name] for name in preferred_names if name in by_name]
    ordered_names = {desp.name for desp in ordered}
    ordered.extend(
        sorted(
            (desp for desp in candidates if desp.name not in ordered_names),
            key=lambda desp: desp.priority,
            reverse=True,
        )
    )
    return ordered


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
