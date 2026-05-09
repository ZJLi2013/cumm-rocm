#!/usr/bin/env python3
"""Parse rocprofv3 kernel_trace.csv and print summary for kernel_0."""
import csv
import sys

f = sys.argv[1]
rows = list(csv.DictReader(open(f)))
kernel_rows = [r for r in rows if r.get("Kernel_Name", "") == "kernel_0"]
if not kernel_rows:
    print("No kernel_0 found")
    sys.exit(0)
durs = [int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in kernel_rows]
avg = sum(durs) / len(durs) / 1000
print(f"kernel_0: n={len(durs)}, avg={avg:.1f}us, min={min(durs)/1000:.1f}us, max={max(durs)/1000:.1f}us")
r0 = kernel_rows[0]
for k in ["Private_Segment_Size", "Group_Segment_Size", "SGPR", "VGPR"]:
    if k in r0:
        print(f"  {k}: {r0[k]}")
