"""mfma_f32_16x16x4f32_n2_ashared_kpipe implicit GEMM family member.

The K-pipe family now uses the AStager-thinned implementation internally:
row LUT / validity is staged once per kv and reused by each BLOCK_K stage.
"""
from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_astager import (
    MFMA_F32_16X16X4F32_N2_ASHARED_KPIPE_ASTAGER_COMPILED_KERNELS
    as MFMA_F32_16X16X4F32_N2_ASHARED_KPIPE_COMPILED_KERNELS,
)
from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_astager import (
    _compile_implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_astager
    as _compile_implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe,
)
from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_astager import (
    implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_astager_forward
    as implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_forward,
)
