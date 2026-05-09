"""Compile crossk kernel variants and dump IR."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["FLYDSL_DUMP_IR"] = "1"
os.environ["FLYDSL_DUMP_DIR"] = "/tmp/crossk_ir"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_crossk import _compile_crossk

bk = int(sys.argv[1]) if len(sys.argv) > 1 else 32
c_in = int(sys.argv[2]) if len(sys.argv) > 2 else 64

print(f"Compiling crossk BK={bk} C_IN={c_in} C_OUT=128 KV=27")
_compile_crossk(c_in, 128, 27, "f32", bk)
print("Done. IR dumped to /tmp/crossk_ir/")
