#!/bin/bash
set -eu

cd /tmp/cumm-rocm

echo "=== Counters: W vec4 ==="
FLYDSL_RUNTIME_ENABLE_CACHE=0 WORKLOAD_DIR=/tmp/kpipe_step4/workloads_wvec4 VARIANTS=direct SHAPES=64x128 ITERS=20 WARMUP=5 INSTALL_ROCPROF_COMPUTE_DEPS=1 \
  bash profile/run_rocprof_compute_kpipe.sh 2>&1 | tail -15

echo "=== Extract and compare ==="
python3 /tmp/extract_counters.py /tmp/kpipe_step4/workloads_wvec4 /tmp/kpipe_step3/workloads_vec4

echo "=== Step 4 counters done ==="
