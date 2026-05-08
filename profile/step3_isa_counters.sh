#!/bin/bash
set -eu

cd /tmp/cumm-rocm

# Patch FlyDSL ast_rewriter
python - <<'PYEOF'
import urllib.request
import flydsl.compiler.ast_rewriter as m
path = m.__file__
url = "https://raw.githubusercontent.com/ZJLi2013/FlyDSL/main/python/flydsl/compiler/ast_rewriter.py"
open(path, "w").write(urllib.request.urlopen(url).read().decode("utf-8"))
print(f"Replaced {path}")
PYEOF

# === Part 1: ISA dump for vec4 (HEAD = 8cdcb5d) ===
echo "=== ISA dump: vec4 HEAD ==="
rm -rf /tmp/kpipe_isa_vec4
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/kpipe_isa_vec4 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  PYTHONPATH=/tmp/cumm-rocm python /tmp/cumm-rocm/profile/profile_kpipe_kernel.py \
    --variant direct --n-active 20000 --c-in 64 --c-out 128 --iters 3 --warmup 1

echo "=== ISA vec4 files ==="
ls -la /tmp/kpipe_isa_vec4/ 2>/dev/null || echo "no dump dir"

# Search for vectorized load/store instructions in ISA
echo "=== ISA: buffer_load patterns ==="
grep -c 'buffer_load_dwordx4\|buffer_load_dwordx2\|buffer_load_dword ' /tmp/kpipe_isa_vec4/*.isa 2>/dev/null || \
  grep -rn 'buffer_load_dwordx4\|buffer_load_dwordx2\|buffer_load_dword ' /tmp/kpipe_isa_vec4/ 2>/dev/null | head -40 || \
  echo "no .isa files found, checking other extensions"

echo "=== ISA: ds_write patterns ==="
grep -c 'ds_write_b128\|ds_write_b64\|ds_write_b32\|ds_write2' /tmp/kpipe_isa_vec4/*.isa 2>/dev/null || \
  grep -rn 'ds_write_b128\|ds_write_b64\|ds_write_b32\|ds_write2' /tmp/kpipe_isa_vec4/ 2>/dev/null | head -40 || \
  echo "checking all files for ds_write"

# List all dumped files to find the right extension
echo "=== All dump files ==="
find /tmp/kpipe_isa_vec4 -type f | head -20

# Try to find ISA content regardless of extension
echo "=== buffer_load in all dump files ==="
grep -rn 'buffer_load' /tmp/kpipe_isa_vec4/ 2>/dev/null | head -30 || echo "no buffer_load found"

echo "=== ds_write in all dump files ==="
grep -rn 'ds_write\|ds_store' /tmp/kpipe_isa_vec4/ 2>/dev/null | head -30 || echo "no ds_write found"

# === Part 2: counters for vec4 (HEAD = 8cdcb5d) ===
echo "=== rocprof-compute counters: vec4 ==="
mkdir -p /tmp/kpipe_step3
WORKLOAD_DIR=/tmp/kpipe_step3/workloads_vec4 VARIANTS=direct SHAPES=64x128 ITERS=20 WARMUP=5 INSTALL_ROCPROF_COMPUTE_DEPS=1 \
  bash /tmp/cumm-rocm/profile/run_rocprof_compute_kpipe.sh 2>&1 | tail -30

# === Part 3: checkout baseline, counters ===
echo "=== checkout baseline 48cec81 ==="
git checkout 48cec81

echo "=== rocprof-compute counters: baseline ==="
FLYDSL_RUNTIME_ENABLE_CACHE=0 WORKLOAD_DIR=/tmp/kpipe_step3/workloads_baseline VARIANTS=direct SHAPES=64x128 ITERS=20 WARMUP=5 INSTALL_ROCPROF_COMPUTE_DEPS=0 \
  bash /tmp/cumm-rocm/profile/run_rocprof_compute_kpipe.sh 2>&1 | tail -30

# === Part 4: restore HEAD ===
git checkout rocm

# === Part 5: extract raw counters for comparison ===
echo "=== Extracting counter CSVs ==="
echo "--- vec4 counters ---"
find /tmp/kpipe_step3/workloads_vec4 -name '*counter_collection.csv' -exec head -5 {} \; 2>/dev/null
echo ""
echo "--- baseline counters ---"
find /tmp/kpipe_step3/workloads_baseline -name '*counter_collection.csv' -exec head -5 {} \; 2>/dev/null

# Extract key metrics from both
echo "=== Key metric comparison ==="
for metric in SQ_INSTS_VMEM TA_BUFFER_READ_WAVEFRONTS_sum SQ_BUSY_CYCLES SQ_INSTS_LDS SQ_LDS_BANK_CONFLICT SQ_INSTS_MFMA; do
    echo "--- $metric ---"
    echo -n "  baseline: "
    grep -rh "$metric" /tmp/kpipe_step3/workloads_baseline/ 2>/dev/null | head -3
    echo -n "  vec4:     "
    grep -rh "$metric" /tmp/kpipe_step3/workloads_vec4/ 2>/dev/null | head -3
done

echo "=== Step 3 complete ==="
