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

echo "=== ISA dump ==="
rm -rf /tmp/kpipe_isa_step6
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/kpipe_isa_step6 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  PYTHONPATH=/tmp/cumm-rocm python /tmp/cumm-rocm/profile/profile_kpipe_kernel.py \
    --variant direct --n-active 20000 --c-in 64 --c-out 128 --iters 1 --warmup 1

echo "=== Dump files ==="
find /tmp/kpipe_isa_step6 -type f | head -20

echo ""
echo "=============================="
echo "=== SALU analysis: div/mod ==="
echo "=============================="

ISA_FILE=$(find /tmp/kpipe_isa_step6 -name '*.isa' -o -name '*.s' | head -1)
if [ -z "$ISA_FILE" ]; then
  echo "No ISA file found, trying all files for s_lshr/s_and patterns"
  ISA_FILE=$(find /tmp/kpipe_isa_step6 -type f | head -1)
fi
echo "Using ISA file: $ISA_FILE"

echo ""
echo "--- 64-bit shift/div instructions ---"
grep -n 's_lshr_b64\|s_lshl_b64\|s_and_b64\|s_div\|s_rem\|v_div\|v_rem\|s_mul_hi' "$ISA_FILE" 2>/dev/null | head -40 || echo "(none)"

echo ""
echo "--- 32-bit shift/mask instructions ---"
grep -cn 's_lshr_b32\|s_lshl_b32\|s_and_b32\|s_ashr' "$ISA_FILE" 2>/dev/null || echo "(none)"

echo ""
echo "--- scalar multiply ---"
grep -cn 's_mul_i32\|s_mulk\|s_mul_hi' "$ISA_FILE" 2>/dev/null || echo "(none)"

echo ""
echo "--- v_mul for address calc ---"
grep -n 'v_mul_lo_u32\|v_mul_hi_u32\|v_mad_u64\|v_mul_u32_u24' "$ISA_FILE" 2>/dev/null | head -40 || echo "(none)"

echo ""
echo "=========================="
echo "=== Instruction counts ==="
echo "=========================="

echo "--- SALU instruction summary ---"
echo -n "  s_lshr_b32: "; grep -c 's_lshr_b32' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  s_lshr_b64: "; grep -c 's_lshr_b64' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  s_lshl_b32: "; grep -c 's_lshl_b32' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  s_lshl_b64: "; grep -c 's_lshl_b64' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  s_and_b32:  "; grep -c 's_and_b32' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  s_and_b64:  "; grep -c 's_and_b64' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  s_mul_i32:  "; grep -c 's_mul_i32' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  s_add_u32:  "; grep -c 's_add_u32' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  s_add_i32:  "; grep -c 's_add_i32' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  s_addc_u32: "; grep -c 's_addc_u32' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  s_sub_u32:  "; grep -c 's_sub_u32' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  s_mov_b32:  "; grep -c 's_mov_b32' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  s_mov_b64:  "; grep -c 's_mov_b64' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  s_cmp:      "; grep -c 's_cmp' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  s_cbranch:  "; grep -c 's_cbranch' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  s_branch:   "; grep -c 's_branch' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  s_waitcnt:  "; grep -c 's_waitcnt' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  s_barrier:  "; grep -c 's_barrier' "$ISA_FILE" 2>/dev/null || echo 0

echo ""
echo "--- VALU instruction summary ---"
echo -n "  v_mul_lo:   "; grep -c 'v_mul_lo' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  v_mul_hi:   "; grep -c 'v_mul_hi' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  v_mad_u64:  "; grep -c 'v_mad_u64' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  v_lshrrev:  "; grep -c 'v_lshrrev' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  v_lshlrev:  "; grep -c 'v_lshlrev' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  v_and_b32:  "; grep -c 'v_and_b32' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  v_add_u32:  "; grep -c 'v_add' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  v_cmp:      "; grep -c 'v_cmp' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  v_cndmask:  "; grep -c 'v_cndmask' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  v_mov_b32:  "; grep -c 'v_mov_b32' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  v_mfma:     "; grep -c 'v_mfma' "$ISA_FILE" 2>/dev/null || echo 0

echo ""
echo "--- Memory instructions ---"
echo -n "  buffer_load_dwordx4: "; grep -c 'buffer_load_dwordx4' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  buffer_load_dword:   "; grep -c 'buffer_load_dword ' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  buffer_store:        "; grep -c 'buffer_store' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  ds_write_b128:  "; grep -c 'ds_write_b128' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  ds_write_b32:   "; grep -c 'ds_write_b32' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  ds_read_b32:    "; grep -c 'ds_read_b32' "$ISA_FILE" 2>/dev/null || echo 0
echo -n "  ds_read_b128:   "; grep -c 'ds_read_b128' "$ISA_FILE" 2>/dev/null || echo 0

echo ""
echo "================================"
echo "=== Context around v_mul/div ==="
echo "================================"
echo "Showing v_mul_lo and surrounding context (address calc patterns):"
grep -n -B2 -A2 'v_mul_lo_u32\|v_mad_u64' "$ISA_FILE" 2>/dev/null | head -80 || echo "(none)"

echo ""
echo "Showing 64-bit add pairs (s_add_u32 + s_addc_u32 = 64-bit add):"
grep -n -A1 's_add_u32.*s[0-9]' "$ISA_FILE" 2>/dev/null | grep -B1 's_addc_u32' | head -40 || echo "(none)"

echo ""
echo "=== Full ISA line count ==="
wc -l "$ISA_FILE"

echo ""
echo "=== Step 6 ISA analysis complete ==="
