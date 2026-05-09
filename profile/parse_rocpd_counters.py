"""Parse rocprofv3 SQLite (rocpd) output and extract PMC counter values.

Usage:
    python parse_rocpd_counters.py /path/to/results_a.db [/path/to/results_b.db]
"""
import sqlite3
import sys
from collections import defaultdict


def extract_counters(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]

    counters = defaultdict(float)
    count = defaultdict(int)

    if "rocpd_info_pmc" in tables:
        cur.execute("SELECT * FROM rocpd_info_pmc")
        cols = [d[0] for d in cur.description]
        for row in cur.fetchall():
            rd = dict(zip(cols, row))
            name = rd.get("counter_name") or rd.get("Counter_Name") or rd.get("name", "")
            val = rd.get("counter_value") or rd.get("Counter_Value") or rd.get("value", 0)
            if name:
                try:
                    counters[name] += float(val)
                    count[name] += 1
                except (ValueError, TypeError):
                    pass

    if not counters and "rocpd_kernel_dispatch" in tables:
        cur.execute("SELECT * FROM rocpd_kernel_dispatch LIMIT 1")
        cols = [d[0] for d in cur.description]
        pmc_cols = [c for c in cols if c.startswith("SQ_") or c.startswith("TCP_") or c.startswith("TCC_")]
        if pmc_cols:
            cur.execute(f"SELECT {','.join(pmc_cols)} FROM rocpd_kernel_dispatch")
            for row in cur.fetchall():
                for i, col in enumerate(pmc_cols):
                    if row[i] is not None:
                        counters[col] += float(row[i])
                        count[col] += 1

    conn.close()

    avg = {}
    for k in counters:
        avg[k] = counters[k] / max(count[k], 1)
    return avg


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_rocpd_counters.py <db_a> [db_b]")
        sys.exit(1)

    db_a = sys.argv[1]
    db_b = sys.argv[2] if len(sys.argv) > 2 else None

    a = extract_counters(db_a)

    if db_b:
        b = extract_counters(db_b)
        all_keys = sorted(set(list(a.keys()) + list(b.keys())))
        label_a = db_a.split("/")[-1].replace("_results.db", "")
        label_b = db_b.split("/")[-1].replace("_results.db", "")
        print(f"\n{'Counter':<35s} {label_a:>15s} {label_b:>15s} {'delta':>10s}")
        print("-" * 80)
        for k in all_keys:
            va = a.get(k, 0)
            vb = b.get(k, 0)
            delta = f"{(vb/va - 1)*100:+.1f}%" if va > 0 else ""
            print(f"{k:<35s} {va:>15.0f} {vb:>15.0f} {delta:>10s}")
    else:
        print(f"\n{'Counter':<35s} {'Value':>15s}")
        print("-" * 55)
        for k in sorted(a.keys()):
            print(f"{k:<35s} {a[k]:>15.0f}")

    print("\n=== Derived Metrics ===")
    for label, d in [("A: " + db_a.split("/")[-1], a)] + (
        [("B: " + db_b.split("/")[-1], b)] if db_b else []
    ):
        vmem_inst = d.get("SQ_INSTS_VMEM", 0)
        lds_inst = d.get("SQ_INSTS_LDS", 0)
        vmem_lat = d.get("SQ_INST_LEVEL_VMEM", 0) / max(vmem_inst, 1)
        lds_lat = d.get("SQ_INST_LEVEL_LDS", 0) / max(lds_inst, 1)
        wait_frac = d.get("SQ_WAIT_ANY", 0) / max(d.get("SQ_WAVE_CYCLES", 1), 1) * 100
        bank_conf = d.get("SQ_LDS_BANK_CONFLICT", 0) / max(lds_inst, 1)
        print(f"\n  {label}:")
        print(f"    VMEM avg latency:   {vmem_lat:.1f} cycles/inst")
        print(f"    LDS avg latency:    {lds_lat:.1f} cycles/inst")
        print(f"    Wait fraction:      {wait_frac:.1f}%")
        print(f"    LDS bank conflicts: {bank_conf:.2f} per inst")
        print(f"    VMEM insts:         {vmem_inst:.0f}")
        print(f"    LDS insts:          {lds_inst:.0f}")
        print(f"    SALU insts:         {d.get('SQ_INSTS_SALU', 0):.0f}")
        print(f"    VALU insts:         {d.get('SQ_INSTS_VALU', 0):.0f}")


if __name__ == "__main__":
    main()
