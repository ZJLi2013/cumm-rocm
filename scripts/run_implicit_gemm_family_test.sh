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

echo "=== Step 1: mfma_f32_16x16x4f32 correctness ==="
python -m pytest test/test_implicit_gemm.py::TestImplicitGemmMfmaF32_16x16x4 -v --tb=long

echo "=== Step 2: mfma_f32_16x16x4f32_n2 correctness ==="
python -m pytest test/test_implicit_gemm.py::TestImplicitGemmMfmaF32_16x16x4N2 -v --tb=long

echo "=== Step 3: mfma_f32_16x16x4f32_n2_ashared correctness ==="
python -m pytest test/test_implicit_gemm.py::TestImplicitGemmMfmaF32_16x16x4N2AShared -v --tb=long

echo "=== Step 4: mfma_f32_32x32x2f32 correctness ==="
python -m pytest test/test_implicit_gemm.py::TestImplicitGemmMfmaF32_32x32x2 -v --tb=long

echo "=== Step 5: scalar_tile regression ==="
python -m pytest test/test_implicit_gemm.py::TestImplicitGemmScalarTile::test_basic_f32 -v --tb=short

echo "=== Step 6: Benchmark current kernel family ==="
python test/bench_implicit_gemm_family.py

echo "=== Done ==="
