# Implicit GEMM Profiling Toolkit

Profiling and benchmarking tools for cumm-rocm implicit GEMM kernels on MI300X (gfx942).

## Quick Start

```bash
# Kernel-only: crossk-PF-XOR vs kpipe vs scalar_tile
PYTHONPATH=/tmp/cumm-rocm python profile/profile_crossk_vs_kpipe.py

# Full latency: dispatch (preprocessing + kernel) vs scalar_tile
PYTHONPATH=/tmp/cumm-rocm python profile/profile_full_latency.py --iters 50

# Sweep crossk configs
PYTHONPATH=/tmp/cumm-rocm python profile/sweep_crossk.py
```

## Standard Shapes

| Shape | N | C_in | C_out | kv | density |
|---|---|---|---|---|---|
| `32x32` | 20000 | 32 | 32 | 27 | 0.3 |
| `32x64` | 20000 | 32 | 64 | 27 | 0.3 |
| `64x128` | 20000 | 64 | 128 | 27 | 0.3 |
| `64x128_50k` | 50000 | 64 | 128 | 27 | 0.3 |

## Benchmark Scripts

### `profile_crossk_vs_kpipe.py` — Primary A/B comparison

Kernel-only latency: `crossk-PF-XOR BK32` vs `kpipe BK16` vs `scalar_tile`.
This is the main benchmark for evaluating kernel performance.

```bash
PYTHONPATH=/tmp/cumm-rocm python profile/profile_crossk_vs_kpipe.py
```

### `profile_full_latency.py` — End-to-end dispatch benchmark

Full latency including host preprocessing (`build_implicit_gemm_mask` +
`_build_active_kv_ids` + `_pack_weights`). Uses `implicit_gemm_forward`
(real dispatch) vs `scalar_tile` baseline.

```bash
PYTHONPATH=/tmp/cumm-rocm python profile/profile_full_latency.py --iters 50
```

### `profile_kpipe_kernel.py` — kpipe-specific kernel driver

Kernel-only latency for `kpipe_bk16` direct/remap variants. Useful for
debugging kpipe-specific regressions.

```bash
PYTHONPATH=/tmp/cumm-rocm python profile/profile_kpipe_kernel.py \
  --variant direct --n-active 20000 --c-in 64 --c-out 128 --iters 200
```

### `sweep_crossk.py` — Multi-config sweep

Sweeps `crossk_bk32` (non-prefetch) vs `kpipe_bk16` across tile sizes
and KV counts. Useful for dispatch threshold tuning.

```bash
PYTHONPATH=/tmp/cumm-rocm python profile/sweep_crossk.py
```

## Profiling Tools (rocprof)

### `run_rocprofv3_kpipe.sh` — Kernel dispatch traces

Records `--kernel-trace --hip-trace --stats` for each shape/variant.

```bash
OUT_DIR=/tmp/kpipe_trace SHAPES="64x128" VARIANTS="direct" \
  bash profile/run_rocprofv3_kpipe.sh
```

### `run_rocprof_compute_kpipe.sh` — Hardware counter collection

Multi-pass counter collection via `rocprof-compute profile` (1000+ counters in ~13 passes).

```bash
WORKLOAD_DIR=/tmp/workloads SHAPES="64x128" VARIANTS="direct" \
  ITERS=20 WARMUP=5 bash profile/run_rocprof_compute_kpipe.sh
```

### `run_kpipe_profile_all.sh` — Full pipeline

Runs: correctness smoke → kernel-only latency → rocprofv3 traces → counters.

```bash
PATCH_FLYDSL=1 OUT_ROOT=/tmp/kpipe_profile bash profile/run_kpipe_profile_all.sh
```

## Counter Utilities

| Script | Purpose |
|--------|---------|
| `extract_counters.py` | Parse rocprof-compute v3 CSV, print key counters, A/B comparison |
| `parse_rocpd_counters.py` | Extract PMC counters from rocpd SQLite databases |
| `inspect_db.py` | Inspect rocpd SQLite schema/tables |
| `parse_trace.py` | Summarize `kernel_trace.csv` dispatch timing |
| `dump_crossk_ir.py` | Compile crossk kernel and dump MLIR IR for inspection |

## Remote Execution

All profiling runs on `smc300x-clt-r4c7-37.cs-clt.dcgpu` in `flydsl-rocm72` Docker container.

```bash
ssh smc300x-clt-r4c7-37.cs-clt.dcgpu \
  "docker exec flydsl-rocm72 bash -c 'cd /tmp/cumm-rocm && git pull origin rocm && \
   PYTHONPATH=/tmp/cumm-rocm python profile/profile_crossk_vs_kpipe.py'"
```

## Bottleneck Analysis Methodology

### Step 1: Latency baseline

Run `profile_crossk_vs_kpipe.py` on standard shapes. Compare median latency
before and after each change. Target: reproducible ±1% on 200 iterations.

### Step 2: Counter collection

Run `rocprof-compute` with `ITERS=20` (13 passes × 20 iters).
Extract via `extract_counters.py`.

### Step 3: Derive bottleneck indicators

| Metric | Formula | Target |
|---|---|---|
| **VMEM avg latency** | `SQ_INST_LEVEL_VMEM / SQ_INSTS_VMEM` | < 100 cycles (with prefetch) |
| **LDS avg latency** | `SQ_INST_LEVEL_LDS / SQ_INSTS_LDS` | < 20 cycles (no bank conflicts) |
| **Wait fraction** | `SQ_WAIT_ANY / SQ_WAVE_CYCLES` | < 30% |
| **VMEM wait share** | `SQ_INST_LEVEL_VMEM / (VMEM + LDS + SMEM)` | Dominant for sparse gather |
| **Bank conflict rate** | `SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS` | < 1.0 |

## Key Counter Reference (gfx942)

### Instruction counts

| Counter | Description |
|---|---|
| `SQ_INSTS_VMEM` | Global memory load/store instructions |
| `SQ_INSTS_LDS` | LDS (shared memory) read/write |
| `SQ_INSTS_MFMA` | Matrix FMA (useful compute) |
| `SQ_INSTS_VALU` | All vector ALU (includes MFMA) |
| `SQ_INSTS_SALU` | Scalar ALU (address calc, control) |

### Wait / latency

| Counter | Description |
|---|---|
| `SQ_INST_LEVEL_VMEM` | Cumulative VMEM wait cycles |
| `SQ_INST_LEVEL_LDS` | Cumulative LDS wait cycles |
| `SQ_WAIT_ANY` | Total cycles in any wait state |
| `SQ_WAVE_CYCLES` | Total wave-cycles (all waves × lifetime) |

### Conflicts / stalls

| Counter | Description |
|---|---|
| `SQ_LDS_BANK_CONFLICT` | LDS bank conflict events |
| `TCP_PENDING_STALL_CYCLES_sum` | TCP stall waiting for cache data |
| `SPI_RA_BAR_CU_FULL_CSN` | Barrier backpressure stall |
