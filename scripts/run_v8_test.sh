#!/usr/bin/env bash
set -ex

echo "=== Setup cumm-rocm ==="
if [ -d /workspace/cumm-rocm ]; then
    cd /workspace/cumm-rocm
    git fetch origin rocm
    git reset --hard origin/rocm
else
    git clone --branch rocm --depth 1 https://github.com/ZJLi2013/cumm-rocm.git /workspace/cumm-rocm
fi

echo "=== Patch FlyDSL ast_rewriter.py ==="
python << 'PYEOF'
import urllib.request
import flydsl.compiler.ast_rewriter as m
path = m.__file__
url = "https://raw.githubusercontent.com/ZJLi2013/FlyDSL/main/python/flydsl/compiler/ast_rewriter.py"
fixed_src = urllib.request.urlopen(url).read().decode('utf-8')
with open(path, 'w') as f:
    f.write(fixed_src)
print(f"Replaced {path}")
PYEOF

echo "=== Install cumm-rocm ==="
cd /workspace/cumm-rocm
pip install -e . --quiet

echo "=== Step 1: V8c correctness ==="
python -m pytest test/test_implicit_gemm.py::TestImplicitGemmV8C -v --tb=long

echo "=== Step 2: V8a regression ==="
python -m pytest test/test_implicit_gemm.py::TestImplicitGemmV8::test_basic_f32 -v --tb=short

echo "=== Step 3: Benchmark (V8a vs V8c) ==="
python test/bench_v8.py

echo "=== Done ==="
