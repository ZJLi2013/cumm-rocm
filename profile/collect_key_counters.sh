#!/usr/bin/env bash
# Collect key bottleneck counters for crossk BK32 and crossk-PF BK32
# via rocprofv3 with targeted counter sets (single pass each).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/s10_counters}"
mkdir -p "$OUT_DIR"

ITERS=20
WARMUP=5

# Counter groups — each group runs in a separate pass
# Group 1: SQ instruction counts
cat > "$OUT_DIR/input_sq_inst.yaml" << 'EOF'
counter:
  - SQ_INSTS_VMEM
  - SQ_INSTS_LDS
  - SQ_INSTS_SALU
  - SQ_INSTS_VALU
  - SQ_INSTS_SMEM
  - SQ_INSTS_FLAT
  - SQ_INSTS_BRANCH
EOF

# Group 2: SQ wait/latency
cat > "$OUT_DIR/input_sq_wait.yaml" << 'EOF'
counter:
  - SQ_INST_LEVEL_VMEM
  - SQ_INST_LEVEL_LDS
  - SQ_INST_LEVEL_SMEM
  - SQ_WAVE_CYCLES
  - SQ_WAIT_ANY
  - SQ_WAIT_INST_ANY
  - SQ_BUSY_CYCLES
EOF

# Group 3: LDS bank conflicts
cat > "$OUT_DIR/input_lds.yaml" << 'EOF'
counter:
  - SQ_LDS_BANK_CONFLICT
  - SQ_INSTS_LDS
  - SQ_ACTIVE_INST_LDS
  - SQ_INST_CYCLES_LDS
EOF

run_one() {
    local kernel=$1 label=$2 yaml=$3 out_prefix=$4
    echo "=== $label: $kernel ==="
    PYTHONPATH="$ROOT" FLYDSL_RUNTIME_ENABLE_CACHE=0 \
      rocprofv3 --counter-input "$yaml" \
        -o "$OUT_DIR/${out_prefix}" \
        -- python "$ROOT/profile/profile_crossk_pf_kernel.py" \
          --kernel "$kernel" --block-k 32 \
          --c-in 64 --c-out 128 --kv 27 \
          --iters "$ITERS" --warmup "$WARMUP" \
      2>&1 | grep -E 'avg_us|counter|Counter'
    echo ""
}

# crossk BK32 baseline
for yaml_name in input_sq_inst input_sq_wait input_lds; do
    run_one "crossk" "crossk BK32 $yaml_name" "$OUT_DIR/$yaml_name.yaml" "crossk_${yaml_name}"
done

# crossk-PF BK32
for yaml_name in input_sq_inst input_sq_wait input_lds; do
    run_one "crossk_pf" "crossk-PF BK32 $yaml_name" "$OUT_DIR/$yaml_name.yaml" "crossk_pf_${yaml_name}"
done

echo "=== Collecting results ==="
echo ""
echo "--- crossk BK32 ---"
for f in "$OUT_DIR"/crossk_input_*.csv; do
    [ -f "$f" ] && echo "  $f" && head -2 "$f"
done

echo ""
echo "--- crossk-PF BK32 ---"
for f in "$OUT_DIR"/crossk_pf_input_*.csv; do
    [ -f "$f" ] && echo "  $f" && head -2 "$f"
done

# Parse and compare
echo ""
echo "=== Parsing counter CSVs ==="
python3 - << 'PYEOF'
import csv, glob, os, sys

def parse_counter_csv(pattern):
    counters = {}
    for f in sorted(glob.glob(pattern)):
        with open(f) as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                name = row.get("Counter Name") or row.get("counter_name", "")
                val = row.get("Value") or row.get("counter_value", "")
                if name and val:
                    try:
                        counters[name] = float(val)
                    except ValueError:
                        pass
    return counters

out = os.environ.get("OUT_DIR", "/tmp/s10_counters")

ck = parse_counter_csv(f"{out}/crossk_input_*/*.csv") or parse_counter_csv(f"{out}/crossk_input_*/*counter*.csv")
pf = parse_counter_csv(f"{out}/crossk_pf_input_*/*.csv") or parse_counter_csv(f"{out}/crossk_pf_input_*/*counter*.csv")

if not ck and not pf:
    # Try v3 format: directory with agent hash
    ck = parse_counter_csv(f"{out}/crossk_input_*/*/*.csv")
    pf = parse_counter_csv(f"{out}/crossk_pf_input_*/*/*.csv")

if not ck:
    print("WARNING: no crossk counters found")
if not pf:
    print("WARNING: no crossk-PF counters found")

all_keys = sorted(set(list(ck.keys()) + list(pf.keys())))
print(f"\n{'Counter':<35s} {'crossk BK32':>15s} {'crossk-PF BK32':>15s} {'delta':>10s}")
print("-" * 80)
for k in all_keys:
    a = ck.get(k, 0)
    b = pf.get(k, 0)
    delta = ""
    if a > 0:
        delta = f"{(b/a - 1)*100:+.1f}%"
    print(f"{k:<35s} {a:>15.0f} {b:>15.0f} {delta:>10s}")

# Derived metrics
print("\n=== Derived Metrics ===")
for label, d in [("crossk BK32", ck), ("crossk-PF BK32", pf)]:
    vmem_lat = d.get("SQ_INST_LEVEL_VMEM", 0) / max(d.get("SQ_INSTS_VMEM", 1), 1)
    lds_lat = d.get("SQ_INST_LEVEL_LDS", 0) / max(d.get("SQ_INSTS_LDS", 1), 1)
    wait_frac = d.get("SQ_WAIT_ANY", 0) / max(d.get("SQ_WAVE_CYCLES", 1), 1) * 100
    bank_conf = d.get("SQ_LDS_BANK_CONFLICT", 0) / max(d.get("SQ_INSTS_LDS", 1), 1)
    print(f"\n  {label}:")
    print(f"    VMEM avg latency:   {vmem_lat:.1f} cycles/inst")
    print(f"    LDS avg latency:    {lds_lat:.1f} cycles/inst")
    print(f"    Wait fraction:      {wait_frac:.1f}%")
    print(f"    LDS bank conflicts: {bank_conf:.2f} per inst")
PYEOF
