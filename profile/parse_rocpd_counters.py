"""Parse rocprofv3 SQLite (rocpd) output and extract PMC counter values.

Usage:
    python parse_rocpd_counters.py /path/to/results_a.db [/path/to/results_b.db]
"""
import sqlite3
import sys
from collections import defaultdict


def _find_table(tables, prefix):
    for t in tables:
        if t.startswith(prefix):
            return t
    return None


def extract_counters(db_path, target_kernel=None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]

    pmc_info_t = _find_table(tables, "rocpd_info_pmc_")
    pmc_event_t = _find_table(tables, "rocpd_pmc_event_")
    kd_t = _find_table(tables, "rocpd_kernel_dispatch_")
    ks_t = _find_table(tables, "rocpd_info_kernel_symbol_")

    if not (pmc_info_t and pmc_event_t and kd_t):
        print(f"WARNING: missing tables in {db_path}")
        return {}

    pmc_names = {}
    cur.execute(f"SELECT id, name FROM {pmc_info_t}")
    for row in cur.fetchall():
        pmc_names[row[0]] = row[1]

    target_event_ids = set()
    if target_kernel and ks_t:
        cur.execute(f"SELECT id, kernel_name FROM {ks_t}")
        target_kid = None
        for row in cur.fetchall():
            if target_kernel in str(row[1]):
                target_kid = row[0]
                break
        if target_kid is not None:
            cur.execute(f"SELECT event_id FROM {kd_t} WHERE kernel_id = ?", (target_kid,))
            target_event_ids = {r[0] for r in cur.fetchall()}
    else:
        cur.execute(f"SELECT event_id FROM {kd_t}")
        target_event_ids = {r[0] for r in cur.fetchall()}

    counters = defaultdict(float)
    count = defaultdict(int)
    cur.execute(f"SELECT event_id, pmc_id, value FROM {pmc_event_t}")
    for eid, pid, val in cur.fetchall():
        if eid in target_event_ids and pid in pmc_names:
            name = pmc_names[pid]
            if val is not None:
                counters[name] += float(val)
                count[name] += 1

    conn.close()

    avg = {}
    for k in counters:
        avg[k] = counters[k] / max(count[k], 1)
    return avg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("db_a", help="Path to first rocpd results.db")
    parser.add_argument("db_b", nargs="?", help="Path to second rocpd results.db (for A/B comparison)")
    parser.add_argument("--kernel", default=None, help="Filter to kernel name substring")
    args = parser.parse_args()

    db_a = args.db_a
    db_b = args.db_b

    a = extract_counters(db_a, target_kernel=args.kernel)

    if db_b:
        b = extract_counters(db_b, target_kernel=args.kernel)
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
