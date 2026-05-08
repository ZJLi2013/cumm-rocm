#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/tmp/kpipe_profile}"
SHAPES="${SHAPES:-32x32 64x128}"
VARIANTS="${VARIANTS:-direct remap}"

cd "$ROOT"

echo "=== Environment ==="
python - <<'PYEOF'
import shutil
import torch
print("torch", torch.__version__)
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
print("rocprofv3", shutil.which("rocprofv3"))
print("rocprof-compute", shutil.which("rocprof-compute"))
PYEOF

echo "=== Correctness smoke ==="
PYTHONPATH="$ROOT" python -m pytest \
  test/test_implicit_gemm.py::TestImplicitGemmDispatchDescriptors \
  test/test_implicit_gemm.py::TestImplicitGemmMfmaF32_16x16x4N2ASharedKPipe \
  -q

echo "=== Warm compile target kernels ==="
for shape in $SHAPES; do
  case "$shape" in
    32x32) C_IN=32; C_OUT=32; N_ACTIVE="${N_ACTIVE_32:-20000}" ;;
    64x128) C_IN=64; C_OUT=128; N_ACTIVE="${N_ACTIVE_64:-20000}" ;;
    *) echo "Unknown shape '$shape'" >&2; exit 2 ;;
  esac
  for variant in $VARIANTS; do
    PYTHONPATH="$ROOT" python "$ROOT/profile/profile_kpipe_kernel.py" \
      --variant "$variant" --n-active "$N_ACTIVE" --c-in "$C_IN" --c-out "$C_OUT" \
      --iters 5 --warmup 2
  done
done

echo "=== rocprofv3 traces ==="
OUT_DIR="$OUT_ROOT/rocprofv3" SHAPES="$SHAPES" VARIANTS="$VARIANTS" \
  bash "$ROOT/profile/run_rocprofv3_kpipe.sh"

echo "=== rocprof-compute counters ==="
WORKLOAD_DIR="$OUT_ROOT/workloads" SHAPES="$SHAPES" VARIANTS="$VARIANTS" \
  bash "$ROOT/profile/run_rocprof_compute_kpipe.sh"

echo "Profile output root: $OUT_ROOT"
