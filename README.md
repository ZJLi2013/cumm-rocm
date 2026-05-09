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
  spconv → FlyDSLGemmTuner  → [FlyDSL hgemm_splitk  (Python → MLIR → ROCDL → GPU)]
         → implicit_gemm     → [FlyDSL MFMA kernels   (Python → MLIR → ROCDL → GPU)]
```

## Kernel Family

cumm-rocm ships two kernel subsystems:

### Dense GEMM (`gemm_tuner.py`)

Drop-in replacement for cumm's `GemmTunerSimple`. Routes to FlyDSL's `hgemm_splitk` / `preshuffle_gemm` for standard matrix multiply.

### Implicit GEMM (`implicit_gemm.py`)

Fused gather→GEMM→scatter kernels for sparse convolution (`spconv`). The dispatch
selects the best kernel based on `dtype`, `C_in`, `C_out`, and runtime `KV` count:

| Kernel | Tile | BLOCK_K | Features | Best For |
|--------|------|---------|----------|----------|
| `crossk_pf_xor_bk32` | 16×32 | 32 | Cross-KV fusion, prefetch, XOR swizzle | C_in≥32, C_out≥32, KV≥15 |
| `kpipe_bk16` | 16×32 | 16 | K-pipelined, A-shared, vec4 loads | C_in≥4, C_out≥32, small C_in or low KV |
| `ashared` | 16×32 | — | A in LDS, MFMA | C_in≤16, C_out≥32 |
| `scalar_tile` | 64×C_out | — | Scalar gather, no MFMA | Universal fallback |

Dispatch logic (automatic, or override via `CUMM_IMPLICIT_GEMM_KERNEL` env var):

```
C_in ≤ 16, C_out ≥ 32:    ashared → kpipe_bk16 → scalar_tile
C_in ≥ 32, C_out % 32 = 0: crossk_pf_xor_bk32 → kpipe_bk16 → scalar_tile
Other:                      scalar_tile
```

`crossk_pf_xor_bk32` has runtime guards (KV≥15, C_in×C_out≥2048) — if not met
it returns `None` and the dispatch automatically falls through to the next candidate.

### Performance (vs scalar_tile baseline, KV=27, MI308X gfx942)

| Config | Kernel-only | Full latency (w/ preprocessing) |
|--------|------------|-------------------------------|
| 64×128, N=50K | −56% | **−37.5%** |
| 64×128, N=20K | −56% | **−34.6%** |
| 32×64 | −50% | **−6.5%** |
| 32×32 (fallback to kpipe) | −62% | **−17.2%** |
| 16×32 | −25% | −0.1% |

## Requirements

- AMD Instinct MI300X/MI308X/MI325X (gfx942) or MI350X (gfx950)
- ROCm 7.1+
- PyTorch with ROCm support
- FlyDSL nightly wheel
- C++/HIP compiler (for `csrc_hip/` mask generation module)

## Installation

```bash
# Install FlyDSL
pip install --extra-index-url https://rocm.frameworks-nightlies.amd.com/whl/gfx942-gfx950/ flydsl

# Install cumm-rocm
pip install -e .

# Build HIP mask generation extension
python setup.py build_ext --inplace
```

## Usage

### Dense GEMM

```python
from cumm.gemm_tuner import FlyDSLGemmTuner

tuner = FlyDSLGemmTuner()
tuner.matmul(c, a, b)           # C = A @ B^T
tuner.matmul(c, a, b, beta=1.0) # accumulate
```

### Implicit GEMM (Sparse Convolution)

```python
from cumm.implicit_gemm import implicit_gemm_forward

# Automatic kernel selection
out = implicit_gemm_forward(features, filters, indice_pairs, indice_pair_num, num_activate_out)
```

```bash
# Force a specific kernel via environment variable
CUMM_IMPLICIT_GEMM_KERNEL=crossk_pf_xor_bk32 python my_script.py
CUMM_IMPLICIT_GEMM_KERNEL=scalar_tile python my_script.py
```

## Testing

```bash
# Full test suite (requires GPU + FlyDSL)
python -m pytest test/test_implicit_gemm.py -v

# Specific kernel family
python -m pytest test/test_implicit_gemm.py::TestImplicitGemmCrossKPrefetch -v
python -m pytest test/test_implicit_gemm.py::TestImplicitGemmDispatch -v

# GEMM tuner
python -m pytest test/test_gemm_tuner.py -v
```

## Benchmarking

```bash
# Kernel-only: crossk-PF-XOR vs kpipe vs scalar_tile
python profile/profile_crossk_vs_kpipe.py

# Full latency: preprocessing + kernel (end-to-end dispatch)
python profile/profile_full_latency.py --iters 50

# Sweep across configs
python profile/sweep_crossk.py
```

## Repository Structure

```
cumm-rocm/
├── cumm/
│   ├── __init__.py                    # Package root (v0.9.0+rocm1)
│   ├── implicit_gemm.py               # Kernel family dispatch & descriptor registry
│   ├── implicit_gemm_common.py         # Shared utilities: mask gen, weight packing
│   ├── implicit_gemm_scalar_tile.py    # Scalar fallback kernel
│   ├── implicit_gemm_mfma_f32_16x16x4f32.py          # Single MFMA 16×16
│   ├── implicit_gemm_mfma_f32_16x16x4f32_n2.py       # 2-wave MFMA
│   ├── implicit_gemm_mfma_f32_16x16x4f32_n2_ashared.py         # A in LDS
│   ├── implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe.py   # K-pipelined
│   ├── implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_db.py # Double-buffered
│   ├── implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_crossk.py  # Cross-KV + prefetch + XOR
│   ├── implicit_gemm_mfma_f32_32x32x2f32.py          # 32×32 MFMA variant
│   ├── gemm_tuner.py                  # Dense GEMM (FlyDSL hgemm_splitk wrapper)
│   └── csrc_hip/                      # C++/HIP mask generation & hash table
├── test/
│   ├── test_implicit_gemm.py          # Comprehensive kernel correctness & dispatch tests
│   ├── test_gemm_tuner.py             # Dense GEMM tuner tests
│   └── bench_implicit_gemm_family.py  # Family-wide benchmark harness
├── profile/                           # Profiling & benchmarking scripts
│   ├── profile_crossk_vs_kpipe.py     # Kernel-only A/B comparison
│   ├── profile_full_latency.py        # End-to-end latency (preprocessing + kernel)
│   └── sweep_crossk.py               # Multi-config sweep
└── scripts/
    └── run_implicit_gemm_family_test.sh
```

## Supported Operations

| Operation | Status | Kernel |
|-----------|--------|--------|
| Implicit GEMM (f32, sparse conv) | ✅ | `crossk_pf_xor_bk32` / `kpipe_bk16` / `scalar_tile` |
| Dense GEMM (fp16/bf16) | ✅ | `hgemm_splitk` |
| Dense GEMM (fp8/int8) | ✅ | `preshuffle_gemm` |
| Split-K GEMM | ✅ | `hgemm_splitk` (split_k=1~16) |
| Fused epilogue (bias+activation) | ✅ | `preshuffle_gemm` |

## Known Limitations

- Implicit GEMM only supports `f32` dtype (f16/bf16 planned).
- `C_out` must be ≥ 32 for MFMA kernels; smaller sizes fall back to `scalar_tile`.
- The `crossk` kernel requires `C_in % 32 == 0` and `C_out % 32 == 0`.
- JIT compilation on first call adds ~1-3s latency; subsequent calls use in-memory cache.
  Set `FLYDSL_RUNTIME_ENABLE_CACHE=1` (default) for persistent disk cache.

## License

Apache-2.0 (same as original cumm)
