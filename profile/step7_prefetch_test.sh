#!/bin/bash
set -eu

cd /tmp/cumm-rocm

echo "=== Pull latest ==="
git fetch origin rocm
git reset --hard origin/rocm
git log --oneline -3

python - <<'PYEOF'
import urllib.request
import flydsl.compiler.ast_rewriter as m
path = m.__file__
url = "https://raw.githubusercontent.com/ZJLi2013/FlyDSL/main/python/flydsl/compiler/ast_rewriter.py"
open(path, "w").write(urllib.request.urlopen(url).read().decode("utf-8"))
print(f"Replaced {path}")
PYEOF

echo "=== Correctness test ==="
FLYDSL_RUNTIME_ENABLE_CACHE=0 PYTHONPATH=/tmp/cumm-rocm python -m pytest \
  /tmp/cumm-rocm/test/test_implicit_gemm.py::TestImplicitGemmMfmaF32_16x16x4N2ASharedKPipe \
  -v --tb=long 2>&1 | tail -60

echo "=== Kernel-only profile: 64x128 ==="
FLYDSL_RUNTIME_ENABLE_CACHE=0 PYTHONPATH=/tmp/cumm-rocm python /tmp/cumm-rocm/profile/profile_kpipe_kernel.py \
  --variant direct --n-active 20000 --c-in 64 --c-out 128 --iters 200 --warmup 10

echo "=== Kernel-only profile: 32x32 ==="
FLYDSL_RUNTIME_ENABLE_CACHE=0 PYTHONPATH=/tmp/cumm-rocm python /tmp/cumm-rocm/profile/profile_kpipe_kernel.py \
  --variant direct --n-active 20000 --c-in 32 --c-out 32 --iters 200 --warmup 10

echo "=== rocprofv3 kernel trace: 64x128 ==="
mkdir -p /tmp/kpipe_step7
FLYDSL_RUNTIME_ENABLE_CACHE=0 rocprofv3 --kernel-trace \
  -o /tmp/kpipe_step7/trace_64x128 \
  python /tmp/cumm-rocm/profile/profile_kpipe_kernel.py \
  --variant direct --n-active 20000 --c-in 64 --c-out 128 --iters 20 --warmup 5
echo "--- trace results ---"
python3 -c "
import csv, sys
f = '/tmp/kpipe_step7/trace_64x128/kernel_trace.csv'
rows = list(csv.DictReader(open(f)))
kernel_rows = [r for r in rows if r.get('Kernel_Name','') == 'kernel_0']
if not kernel_rows:
    print('No kernel_0 found'); sys.exit(0)
durs = [int(r['End_Timestamp']) - int(r['Start_Timestamp']) for r in kernel_rows]
print(f'kernel_0: n={len(durs)}, avg={sum(durs)/len(durs)/1000:.1f}us, min={min(durs)/1000:.1f}us, max={max(durs)/1000:.1f}us')
r0 = kernel_rows[0]
for k in ['Private_Segment_Size','Group_Segment_Size','SGPR','VGPR']:
    if k in r0: print(f'  {k}: {r0[k]}')
"

echo "=== Counters: prefetch ==="
FLYDSL_RUNTIME_ENABLE_CACHE=0 WORKLOAD_DIR=/tmp/kpipe_step7/workloads_prefetch VARIANTS=direct SHAPES=64x128 ITERS=20 WARMUP=5 INSTALL_ROCPROF_COMPUTE_DEPS=1 \
  bash /tmp/cumm-rocm/profile/run_rocprof_compute_kpipe.sh 2>&1 | tail -20

echo "=== Extract counters (prefetch vs Step5 baseline) ==="
python3 /tmp/cumm-rocm/profile/extract_counters.py \
  /tmp/kpipe_step7/workloads_prefetch \
  /tmp/kpipe_step5/workloads_padding \
  2>&1 | grep -E '===|Key|SQ_INSTS_VMEM|TA_BUFFER_READ|SQ_BUSY_CYCLES|SQ_INSTS_LDS|SQ_LDS_BANK|SQ_INSTS_SALU|SQ_INSTS_VALU|SQ_INSTS_MFMA|SQ_WAIT|SQ_INST_LEVEL|VGPR|Workgroup'

echo "=== Step 7 complete ==="
