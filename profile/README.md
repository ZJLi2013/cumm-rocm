# K-pipe Profiling

This directory contains reproducible profiling entry points for comparing
`mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16` direct epilogue against
`mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_remap`.

The default shapes match the current optimization notes:

- `32x32`: `N=20000, C_in=32, C_out=32, kv=27, density=0.3`
- `64x128`: `N=20000, C_in=64, C_out=128, kv=27, density=0.3`

## Quick Run

Inside the ROCm/FlyDSL container:

```bash
cd /tmp/cumm-rocm
PATCH_FLYDSL=1 bash profile/run_kpipe_profile_all.sh
```

Outputs are written under `/tmp/kpipe_profile` by default.
`run_rocprof_compute_kpipe.sh` installs
`/opt/rocm-7.2.0/libexec/rocprofiler-compute/requirements.txt` by default when
the requirements file exists. Set `INSTALL_ROCPROF_COMPUTE_DEPS=0` to skip this.

## Individual Steps

Kernel-only smoke:

```bash
PYTHONPATH=/tmp/cumm-rocm python profile/profile_kpipe_kernel.py \
  --variant direct --n-active 20000 --c-in 64 --c-out 128 --iters 20
```

Trace kernel dispatches with `rocprofv3`:

```bash
bash profile/run_rocprofv3_kpipe.sh
```

Collect counters with `rocprof-compute`:

```bash
bash profile/run_rocprof_compute_kpipe.sh
```

After `rocprof-compute profile`, list supported metrics for the actual output
directory and architecture before selecting sections:

```bash
rocprof-compute analyze -p workloads/<run>/<arch>/ --list-metrics gfx942
```

Focus areas:

- LDS instructions, bandwidth, and bank conflicts
- Global/L2/HBM write traffic and store transaction behavior
- Barrier/wait/scheduler stall metrics
- Occupancy, VGPR, and LDS footprint
- MFMA/VALU utilization
