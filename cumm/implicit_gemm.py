"""FlyDSL Implicit GEMM — backward-compatible re-export layer.

Each version is implemented in its own module:
  - implicit_gemm_v3.py: Pair-scan kernel
  - implicit_gemm_v4.py: LUT-based gather
  - implicit_gemm_v5.py: True K-fused (LDS accumulator)
  - implicit_gemm_v6.py: Output tile + register accumulator (upcoming)

This file re-exports all public APIs so existing imports continue to work.
"""

# Common utilities
from cumm.implicit_gemm_common import (
    _get_hip_module,
    preprocess_pairs,
)

# V3: pair-scan
from cumm.implicit_gemm_v3 import (
    _V3_COMPILED_KERNELS as _COMPILED_KERNELS,
    _compile_implicit_gemm_v3 as _compile_implicit_gemm,
    implicit_gemm_v3_forward as implicit_gemm_forward,
)

# V4: LUT-based gather
from cumm.implicit_gemm_v4 import (
    _V4_COMPILED_KERNELS,
    _compile_implicit_gemm_v4,
    implicit_gemm_v4_forward,
)

# V5: K-fused LDS accumulator
from cumm.implicit_gemm_v5 import (
    _V5_COMPILED_KERNELS,
    _compile_implicit_gemm_v5,
    implicit_gemm_v5_forward,
)

# V6: Output-tiled (column tiling)
from cumm.implicit_gemm_v6 import (
    _V6_COMPILED_KERNELS,
    _compile_implicit_gemm_v6,
    implicit_gemm_v6_forward,
)

# V7: register accumulator scalar baseline
from cumm.implicit_gemm_v7 import (
    _V7_COMPILED_KERNELS,
    _compile_implicit_gemm_v7,
    implicit_gemm_v7_forward,
)

# V8a: tile-owned scalar baseline (kept for benchmark comparison)
from cumm.implicit_gemm_v8a import (
    _V8A_COMPILED_KERNELS,
    _compile_implicit_gemm_v8a,
    implicit_gemm_v8a_forward,
)

# V8b: tile-owned scalar baseline with A/B LDS tiles
from cumm.implicit_gemm_v8 import (
    _V8_COMPILED_KERNELS,
    _compile_implicit_gemm_v8,
    implicit_gemm_v8_forward,
)
