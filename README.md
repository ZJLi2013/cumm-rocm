# cumm-rocm

ROCm port of [cumm](https://github.com/FindDefinition/cumm) (CUda Matrix Multiply) — the GEMM backend for [spconv](https://github.com/traveller59/spconv).

## Architecture

The original cumm uses Python metaprogramming (pccm) to generate CUDA C++ kernels at build time.
This ROCm port **replaces the entire codegen + nvcc pipeline** with [FlyDSL](https://github.com/ROCm/FlyDSL),
a Python-native kernel DSL that JIT-compiles directly to AMD GPU instructions via MLIR.

```
Original cumm (NVIDIA):
  spconv → GemmTunerSimple → [pccm codegen → .cu → nvcc → CUDA kernel]

cumm-rocm (AMD):
  spconv → FlyDSLGemmTuner → [FlyDSL hgemm_splitk (Python → MLIR → ROCDL → GPU)]
```

## Requirements

- AMD Instinct MI300X/MI308X/MI325X (gfx942) or MI350X (gfx950)
- ROCm 7.1+
- PyTorch with ROCm support
- FlyDSL nightly wheel

## Installation

```bash
# Install FlyDSL
pip install --extra-index-url https://rocm.frameworks-nightlies.amd.com/whl/gfx942-gfx950/ flydsl

# Install cumm-rocm
pip install -e .
```

## Usage

```python
from cumm.gemm_tuner import FlyDSLGemmTuner

tuner = FlyDSLGemmTuner()

# C = A @ B^T  (standard GEMM, same interface as cumm GemmTunerSimple)
tuner.matmul(c, a, b)

# With accumulation (beta=1.0, for spconv multi-KV-position accumulation)
tuner.matmul(c, a, b, beta=1.0)
```

## Supported Operations

| Operation | Status | Kernel |
|-----------|--------|--------|
| Dense GEMM (fp16/bf16) | ✅ | `hgemm_splitk` |
| Dense GEMM (fp8/int8) | ✅ | `preshuffle_gemm` |
| Split-K GEMM | ✅ | `hgemm_splitk` (split_k=1~16) |
| Fused epilogue (bias+activation) | ✅ | `preshuffle_gemm` |

## License

Apache-2.0 (same as original cumm)
