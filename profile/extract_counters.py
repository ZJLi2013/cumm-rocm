#!/usr/bin/env python3
"""Extract raw counter values for kernel_0 from rocprof-compute v3 output.

v3 CSV format: each file is one pmc pass, each row has Counter_Name + Counter_Value
for a single dispatch. We aggregate across all pass files to build a full counter map.
"""
import csv
import glob
import os
import sys


def extract_kernel_counters(workload_dir, label):
    files = sorted(glob.glob(os.path.join(workload_dir, "out/pmc_*/7d*/*_counter_collection.csv")))
    if not files:
        print(f"{label}: no counter CSV found in {workload_dir}")
        return

    counters = {}
    kernel_info = {}
    for f in files:
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                kname = row.get("Kernel_Name", "")
                if kname != "kernel_0":
                    continue
                cname = row.get("Counter_Name", "")
                cval = row.get("Counter_Value", "")
                if cname:
                    counters[cname] = cval
                if not kernel_info:
                    kernel_info = {
                        "Workgroup_Size": row.get("Workgroup_Size", "?"),
                        "LDS_Block_Size": row.get("LDS_Block_Size", "?"),
                        "VGPR_Count": row.get("VGPR_Count", "?"),
                        "Accum_VGPR_Count": row.get("Accum_VGPR_Count", "?"),
                        "SGPR_Count": row.get("SGPR_Count", "?"),
                        "Grid_Size": row.get("Grid_Size", "?"),
                    }

    if not counters:
        print(f"{label}: no kernel_0 rows found in {len(files)} CSV files")
        return

    print(f"=== {label} ({len(counters)} counters from {len(files)} passes) ===")
    for k, v in sorted(kernel_info.items()):
        print(f"  {k}: {v}")

    targets = [
        "SQ_INSTS_VMEM", "SQ_INSTS_LDS", "SQ_INSTS_MFMA",
        "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_SMEM",
        "SQ_LDS_BANK_CONFLICT",
        "SQ_BUSY_CYCLES", "SQ_BUSY_CU_CYCLES",
        "SQ_VALU_MFMA_BUSY_CYCLES",
        "TA_BUFFER_READ_WAVEFRONTS_sum", "TA_BUFFER_WRITE_WAVEFRONTS_sum",
        "SQ_WAVES",
    ]
    print("  --- Key Counters ---")
    for t in targets:
        if t in counters:
            print(f"  {t}: {counters[t]}")

    print("  --- All Available ---")
    for k in sorted(counters.keys()):
        if k not in [t for t in targets]:
            print(f"  {k}: {counters[k]}")


if __name__ == "__main__":
    vec4_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kpipe_step3/workloads_vec4"
    base_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/kpipe_step3/workloads_baseline"

    extract_kernel_counters(vec4_dir, "vec4 (8cdcb5d)")
    print()
    extract_kernel_counters(base_dir, "baseline (48cec81)")
