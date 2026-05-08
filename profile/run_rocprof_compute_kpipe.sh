#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKLOAD_DIR="${WORKLOAD_DIR:-$ROOT/workloads}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"
SHAPES="${SHAPES:-32x32 64x128}"
VARIANTS="${VARIANTS:-direct remap}"

cd "$ROOT"
mkdir -p "$WORKLOAD_DIR"

if [[ "${PATCH_FLYDSL:-0}" == "1" ]]; then
python - <<'PYEOF'
import urllib.request
import flydsl.compiler.ast_rewriter as m
path = m.__file__
url = "https://raw.githubusercontent.com/ZJLi2013/FlyDSL/main/python/flydsl/compiler/ast_rewriter.py"
open(path, "w").write(urllib.request.urlopen(url).read().decode("utf-8"))
print(f"Replaced {path}")
PYEOF
fi

for shape in $SHAPES; do
  case "$shape" in
    32x32)
      N_ACTIVE="${N_ACTIVE_32:-20000}"
      C_IN=32
      C_OUT=32
      ;;
    64x128)
      N_ACTIVE="${N_ACTIVE_64:-20000}"
      C_IN=64
      C_OUT=128
      ;;
    *)
      echo "Unknown shape '$shape'. Use 32x32 or 64x128." >&2
      exit 2
      ;;
  esac

  for variant in $VARIANTS; do
    run_name="kpipe_${shape}_${variant}"
    echo "=== rocprof-compute profile: $run_name ==="
    PYTHONPATH="$ROOT" rocprof-compute profile \
      -n "$run_name" --path "$WORKLOAD_DIR" --no-roof \
      -- python "$ROOT/profile/profile_kpipe_kernel.py" \
        --variant "$variant" \
        --n-active "$N_ACTIVE" \
        --c-in "$C_IN" \
        --c-out "$C_OUT" \
        --iters "$ITERS" \
        --warmup "$WARMUP" \
      2>&1 | tee "$WORKLOAD_DIR/${run_name}_profile.log"
  done
done

echo "rocprof-compute workloads: $WORKLOAD_DIR"
echo "Next: inspect available metrics with:"
echo "  rocprof-compute analyze -p $WORKLOAD_DIR/<run>/<arch>/ --list-metrics gfx942"
echo "Then analyze per-kernel/dispatch, for example:"
echo "  rocprof-compute analyze -p $WORKLOAD_DIR/<run>/<arch>/ -n per_kernel --dispatch <target_dispatch>"
