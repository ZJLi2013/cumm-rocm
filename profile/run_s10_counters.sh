#!/usr/bin/env bash
# Run targeted rocprofv3 counter collection for crossk vs crossk-PF comparison.
# Each counter group runs as a separate rocprofv3 invocation.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"
PROF="profile/profile_crossk_pf_kernel.py"
COMMON="--block-k 32 --c-in 64 --c-out 128 --kv 27 --iters $ITERS --warmup $WARMUP"

echo "pmc: SQ_INSTS_VMEM SQ_INSTS_LDS SQ_INSTS_SALU SQ_INSTS_VALU" > /tmp/s10_g1.txt
echo "pmc: SQ_INST_LEVEL_VMEM SQ_INST_LEVEL_LDS SQ_WAVE_CYCLES SQ_WAIT_ANY" > /tmp/s10_g2.txt
echo "pmc: SQ_LDS_BANK_CONFLICT" > /tmp/s10_g3.txt

run_groups() {
    local kernel=$1 prefix=$2
    for g in 1 2 3; do
        echo "=== $prefix group $g ==="
        PYTHONPATH="$ROOT" FLYDSL_RUNTIME_ENABLE_CACHE=0 \
          rocprofv3 -i /tmp/s10_g${g}.txt -o /tmp/s10_${prefix}_g${g} \
          -- python "$ROOT/$PROF" --kernel "$kernel" $COMMON 2>&1 | tail -3
        echo ""
    done
}

run_groups "crossk" "ck"
run_groups "crossk_pf" "pf"

echo "=== A/B comparison ==="
echo ""
echo "--- crossk BK32 ---"
python "$ROOT/profile/parse_rocpd_counters.py" /tmp/s10_ck_g1_results.db
python "$ROOT/profile/parse_rocpd_counters.py" /tmp/s10_ck_g2_results.db
python "$ROOT/profile/parse_rocpd_counters.py" /tmp/s10_ck_g3_results.db

echo ""
echo "--- crossk-PF BK32 ---"
python "$ROOT/profile/parse_rocpd_counters.py" /tmp/s10_pf_g1_results.db
python "$ROOT/profile/parse_rocpd_counters.py" /tmp/s10_pf_g2_results.db
python "$ROOT/profile/parse_rocpd_counters.py" /tmp/s10_pf_g3_results.db

echo ""
echo "=== Done ==="
