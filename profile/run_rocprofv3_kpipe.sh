#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/kpipe_profile/rocprofv3}"
ITERS="${ITERS:-200}"
WARMUP="${WARMUP:-10}"
SHAPES="${SHAPES:-32x32 64x128}"
VARIANTS="${VARIANTS:-direct remap}"

mkdir -p "$OUT_DIR"
cd "$ROOT"

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
    run_dir="$OUT_DIR/$run_name"
    mkdir -p "$run_dir"
    echo "=== rocprofv3 trace: $run_name ==="
    PYTHONPATH="$ROOT" rocprofv3 \
      --kernel-trace --hip-trace --stats --output-format csv \
      -o "$run_dir/rocprof" \
      -- python "$ROOT/profile/profile_kpipe_kernel.py" \
        --variant "$variant" \
        --n-active "$N_ACTIVE" \
        --c-in "$C_IN" \
        --c-out "$C_OUT" \
        --iters "$ITERS" \
        --warmup "$WARMUP" \
      2>&1 | tee "$run_dir/stdout.log"
  done
done

echo "rocprofv3 output: $OUT_DIR"
