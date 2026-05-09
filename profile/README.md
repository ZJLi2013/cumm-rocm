# K-pipe Profiling

Profiling toolkit for `implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe`
kernel optimization on MI300X (gfx942).

## Standard Shapes

| Shape | N | C_in | C_out | kv | density |
|---|---|---|---|---|---|
| `32x32` | 20000 | 32 | 32 | 27 | 0.3 |
| `64x128` | 20000 | 64 | 128 | 27 | 0.3 |

## Tooling

### 1. `profile_kpipe_kernel.py` — Kernel-only latency

Bypasses Python overhead: compiles once, then repeatedly launches the cached
FlyDSL kernel with prebuilt mask/LUT/packed weights.

```bash
PYTHONPATH=/tmp/cumm-rocm python profile/profile_kpipe_kernel.py \
  --variant direct --n-active 20000 --c-in 64 --c-out 128 --iters 200 --warmup 10
```

Output: median/mean/min latency (us) over `--iters` dispatches.

### 2. `run_rocprofv3_kpipe.sh` — Kernel dispatch traces

Records `--kernel-trace --hip-trace --stats` for each shape/variant combination.
Produces CSV with per-dispatch Start/End timestamps, SGPR/VGPR/LDS counts.

```bash
OUT_DIR=/tmp/kpipe_trace SHAPES="64x128" VARIANTS="direct" \
  bash profile/run_rocprofv3_kpipe.sh
```

Post-process with `parse_trace.py`:

```bash
python profile/parse_trace.py /tmp/kpipe_trace/kpipe_64x128_direct/rocprof/kernel_trace.csv
```

### 3. `run_rocprof_compute_kpipe.sh` — Hardware counter collection

Multi-pass counter collection via `rocprof-compute profile`. Collects all SQ,
TCP, TCC, TA, TD, SPI counters (typically 1000+ counters in 13 passes).

```bash
WORKLOAD_DIR=/tmp/workloads SHAPES="64x128" VARIANTS="direct" \
  ITERS=20 WARMUP=5 \
  bash profile/run_rocprof_compute_kpipe.sh
```

### 4. `extract_counters.py` — Counter extraction & comparison

Parses rocprof-compute v3 CSV output (one-counter-per-row format) and prints
key counters plus full dump. Supports A/B comparison.

```bash
python profile/extract_counters.py <workload_dir_A> [workload_dir_B]
```

Filter specific counters:

```bash
python profile/extract_counters.py /tmp/workloads | grep -E 'SQ_INST_LEVEL|SQ_WAIT|SQ_LDS_BANK'
```

### 5. `run_kpipe_profile_all.sh` — Full pipeline

Runs correctness smoke → kernel-only latency → rocprofv3 traces →
rocprof-compute counters for all shapes/variants in one shot.

```bash
PATCH_FLYDSL=1 OUT_ROOT=/tmp/kpipe_profile bash profile/run_kpipe_profile_all.sh
```

### 6. Step scripts (`step3_*.sh`, `step4_*.sh`, `step5_*.sh`)

Per-experiment scripts that combine git pull + FlyDSL patch + correctness test +
kernel-only profile + counter collection. Template for new experiments:

```bash
# Inside Docker container:
bash /tmp/cumm-rocm/profile/step5_lds_padding_test.sh 2>&1 | tee /tmp/step5_output.log
```

## Remote Execution

All profiling runs on `smc300x-clt-r4c7-37.cs-clt.dcgpu` inside
`flydsl-rocm72:latest` Docker container.

```bash
# Start container (if not running):
ssh smc300x-clt-r4c7-37.cs-clt.dcgpu \
  "docker run -d --name flydsl-rocm72 --device=/dev/kfd --device=/dev/dri \
   --group-add video -v /tmp:/tmp flydsl-rocm72:latest sleep infinity"

# Execute a script:
ssh smc300x-clt-r4c7-37.cs-clt.dcgpu \
  "docker exec flydsl-rocm72 bash /tmp/cumm-rocm/profile/step5_lds_padding_test.sh"
```

FlyDSL ast_rewriter patch (needed when container FlyDSL version is older):

```bash
python -c "
import urllib.request; import flydsl.compiler.ast_rewriter as m
url = 'https://raw.githubusercontent.com/ZJLi2013/FlyDSL/main/python/flydsl/compiler/ast_rewriter.py'
open(m.__file__, 'w').write(urllib.request.urlopen(url).read().decode('utf-8'))
"
```

## Bottleneck Analysis Methodology

### Step 1: Latency baseline

Run `profile_kpipe_kernel.py` on standard shapes. Compare median latency before
and after each change. Target: reproducible ±1% on 200 iterations.

### Step 2: Counter collection

Run `rocprof-compute` with `ITERS=20` (fewer iterations to keep multi-pass
overhead manageable; 13 passes × 20 iters). Extract via `extract_counters.py`.

### Step 3: Derive bottleneck indicators

Key derived metrics from raw counters:

| Metric | Formula | What it tells you |
|---|---|---|
| **VMEM avg latency** | `SQ_INST_LEVEL_VMEM / SQ_INSTS_VMEM` | Cycles per VMEM instruction (target: < 100 with prefetch) |
| **LDS avg latency** | `SQ_INST_LEVEL_LDS / SQ_INSTS_LDS` | Cycles per LDS instruction (bank conflicts inflate this) |
| **Wait fraction** | `SQ_WAIT_ANY / SQ_WAVE_CYCLES` | % of wave lifetime spent waiting (target: < 30%) |
| **VMEM wait share** | `SQ_INST_LEVEL_VMEM / (VMEM + LDS + SMEM)` | VMEM's fraction of total instruction-level wait |
| **Overhead ratio** | `(non-MFMA VALU + SALU + LDS + BRANCH) / MFMA` | Overhead per useful MFMA (target: < 3x) |
| **Bank conflict rate** | `SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS` | Avg conflicts per LDS inst (target: < 1.0) |
| **SALU effective cost** | `SQ_INSTS_SALU × 4` (4 SIMDs share 1 scalar unit) | Effective SALU cycles per SIMD, compare with MFMA cycles |

### Step 4: Prioritize

Rank bottlenecks by wait contribution and feasibility:

| Priority | Criterion |
|---|---|
| P0 | Largest wait contributor AND has known fix |
| P1 | Second largest OR addresses instruction overhead |
| P2+ | Diminishing returns or monitoring |

### Step 5: Implement → profile → compare

For each optimization step:
1. Push change to remote (git push → docker exec git pull)
2. Correctness: `pytest test/test_implicit_gemm.py::*KPipe -v`
3. Latency: `profile_kpipe_kernel.py` on both shapes
4. Counters: `rocprof-compute` → `extract_counters.py`
5. Update `gemm_backend.md` with results, derived metrics, and conclusions

## Key Counter Reference (gfx942)

### Instruction counts

| Counter | Description |
|---|---|
| `SQ_INSTS_VMEM` | Global memory load/store instructions |
| `SQ_INSTS_LDS` | LDS (shared memory) read/write instructions |
| `SQ_INSTS_MFMA` | Matrix FMA instructions (useful compute) |
| `SQ_INSTS_VALU` | All vector ALU instructions (includes MFMA) |
| `SQ_INSTS_SALU` | Scalar ALU instructions (address calc, control flow) |
| `SQ_INSTS_BRANCH` | Branch instructions |

### Wait / latency

| Counter | Description |
|---|---|
| `SQ_INST_LEVEL_VMEM` | Cumulative VMEM wait cycles across all instructions |
| `SQ_INST_LEVEL_LDS` | Cumulative LDS wait cycles |
| `SQ_WAIT_ANY` | Total cycles waves spent in any wait state |
| `SQ_WAVE_CYCLES` | Total wave-cycles (all waves × lifetime) |
| `SQ_WAIT_INST_ANY` | Cycles waiting specifically on instruction completion |

### Conflicts / stalls

| Counter | Description |
|---|---|
| `SQ_LDS_BANK_CONFLICT` | Total LDS bank conflict events |
| `TCP_PENDING_STALL_CYCLES_sum` | TCP stall waiting for cache data |
| `TD_TC_STALL_sum` | Texture unit stall on cache |
| `SPI_RA_BAR_CU_FULL_CSN` | Stall due to barrier backpressure |
| `SPI_RA_WAVE_SIMD_FULL_CSN` | Stall due to SIMD occupancy full |

### Resources

| Counter | Description |
|---|---|
| `VGPR_Count` | Vector GPRs per wave (affects occupancy) |
| `SGPR_Count` | Scalar GPRs per wave |
| `LDS_Block_Size` | LDS bytes per workgroup |
| `SQ_WAVES` | Total waves dispatched |

## Optimization History

| Step | Commit | Change | 64×128 Latency | Cumulative |
|---|---|---|---:|---:|
| Baseline | `48cec81` | — | 982.4 us | — |
| Step 1-2: A vec4 | `8cdcb5d` | `buffer_load_dwordx4` + auto `ds_write_b128` | 952.0 us | -3.1% |
| Step 3: ISA validation | — | Counter analysis, no code change | — | — |
| Step 4: W vec4 | `ea11444` | W-side `buffer_load_dwordx4` | 944.4 us | -3.8% |
| Step 5: A LDS pad | `ad37bd7` | A stride 16→20, bank conflict 8-way→2-way | 679.8 us | **-30.8%** |
| Step 6: SALU cleanup | — | (planned) div/mod → shift/mask | — | — |
| Step 7: Prefetch | — | (planned) A/W prefetch to next c_block | — | — |
