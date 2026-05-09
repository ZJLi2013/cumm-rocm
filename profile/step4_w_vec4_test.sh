#!/bin/bash
set -eu

cd /tmp/cumm-rocm

echo "=== Pull latest ==="
git fetch origin rocm
git reset --hard origin/rocm
git log --oneline -3

# Patch FlyDSL ast_rewriter
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
  -v --tb=long 2>&1 | tail -40

echo "=== Kernel-only profile: 64x128 ==="
FLYDSL_RUNTIME_ENABLE_CACHE=0 PYTHONPATH=/tmp/cumm-rocm python /tmp/cumm-rocm/profile/profile_kpipe_kernel.py \
  --variant direct --n-active 20000 --c-in 64 --c-out 128 --iters 200 --warmup 10

echo "=== Kernel-only profile: 32x32 ==="
FLYDSL_RUNTIME_ENABLE_CACHE=0 PYTHONPATH=/tmp/cumm-rocm python /tmp/cumm-rocm/profile/profile_kpipe_kernel.py \
  --variant direct --n-active 20000 --c-in 32 --c-out 32 --iters 200 --warmup 10

echo "=== Counters: W vec4 ==="
mkdir -p /tmp/kpipe_step4
FLYDSL_RUNTIME_ENABLE_CACHE=0 WORKLOAD_DIR=/tmp/kpipe_step4/workloads_wvec4 VARIANTS=direct SHAPES=64x128 ITERS=20 WARMUP=5 INSTALL_ROCPROF_COMPUTE_DEPS=0 \
  bash /tmp/cumm-rocm/profile/run_rocprof_compute_kpipe.sh 2>&1 | tail -15

echo "=== Extract counters ==="
python3 /tmp/extract_counters.py /tmp/kpipe_step4/workloads_wvec4 /tmp/kpipe_step3/workloads_vec4 2>&1 | grep -E '===|Key|SQ_INSTS_VMEM|TA_BUFFER_READ|SQ_BUSY_CYCLES|SQ_INSTS_LDS|SQ_LDS_BANK|SQ_INSTS_SALU|SQ_INSTS_VALU|SQ_INSTS_MFMA|VGPR|Workgroup'

echo "=== Step 4 complete ==="
