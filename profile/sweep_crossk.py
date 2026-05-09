"""Step 9: comprehensive crossk sweep across C_IN, C_OUT, BLOCK_K configs."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cumm.implicit_gemm_common import _get_hip_module, _pack_weights
from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe import (
    MFMA_F32_16X16X4F32_N2_ASHARED_KPIPE_COMPILED_KERNELS,
    implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward,
)
from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_crossk import (
    CROSSK_COMPILED_KERNELS,
    _build_active_kv_ids,
    implicit_gemm_crossk_forward,
)


def make_subm_pairs(n_active, kv, density, device):
    nhot_per_kv = []
    max_nhot = 0
    for k in range(kv):
        if k == kv // 2:
            nhot = n_active
        else:
            nhot = int(n_active * density)
        nhot_per_kv.append(nhot)
        max_nhot = max(max_nhot, nhot)
    indice_pairs = -torch.ones(kv, 2, max_nhot, dtype=torch.int32, device=device)
    indice_pair_num = torch.zeros(kv, dtype=torch.int32, device=device)
    for k, nhot in enumerate(nhot_per_kv):
        indice_pair_num[k] = nhot
        inp = torch.randperm(n_active, device=device, dtype=torch.int32)[:nhot]
        out = torch.randperm(n_active, device=device, dtype=torch.int32)[:nhot]
        if k == kv // 2:
            out = inp.clone()
        indice_pairs[k, 0, :nhot] = inp
        indice_pairs[k, 1, :nhot] = out
    return indice_pairs, indice_pair_num


def bench_kernel(fn, warmup=10, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / max(iters, 1)


CONFIGS = [
    # (c_in, c_out, kv, n_active, density)
    (32, 32, 27, 20000, 0.3),
    (32, 64, 27, 20000, 0.3),
    (64, 64, 27, 20000, 0.3),
    (64, 128, 27, 20000, 0.3),
    (128, 128, 27, 20000, 0.3),
    (128, 256, 27, 20000, 0.3),
    (32, 32, 3, 20000, 0.5),
    (64, 128, 3, 20000, 0.5),
]


def run_config(c_in, c_out, kv, n_active, density):
    torch.manual_seed(0)
    device = "cuda"
    block_m = 16

    features = torch.randn(n_active, c_in, dtype=torch.float32, device=device) * 0.1
    filters = torch.randn(kv, c_in, c_out, dtype=torch.float32, device=device) * 0.1
    ip, ipn = make_subm_pairs(n_active, kv, density, device)

    hip = _get_hip_module()
    assert hip is not None
    num_tiles = (n_active + block_m - 1) // block_m
    _, _, _, mask, _, _, inp_row_lut = hip.build_implicit_gemm_mask(ip, ipn, n_active, block_m)
    features_c = features.contiguous()
    weights_packed = _pack_weights(filters, 16)
    lut_flat = inp_row_lut.reshape(-1).contiguous()
    mask_flat = mask.reshape(-1).contiguous()
    stream = torch.cuda.current_stream()

    # --- kpipe BK16 baseline ---
    out_kpipe = implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward(
        features, filters, ip, ipn, n_active
    )
    if out_kpipe is None:
        print(f"  kpipe BK16: COMPILE FAILED")
        return
    torch.cuda.synchronize()

    kpipe_key = (f"mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16", c_in, c_out, kv, "f32", 16, "direct")
    kpipe_launch, _ = MFMA_F32_16X16X4F32_N2_ASHARED_KPIPE_COMPILED_KERNELS[kpipe_key]
    out_buf = torch.zeros(n_active, c_out, dtype=torch.float32, device=device)

    kpipe_us = bench_kernel(
        lambda: kpipe_launch(features_c, weights_packed, out_buf, lut_flat, mask_flat, num_tiles, n_active, stream)
    )
    print(f"  {'kpipe BK16':30s}  {kpipe_us:8.1f} us  (baseline)")

    # --- crossk preprocessing ---
    mask_2d = mask.reshape(num_tiles, kv)
    active_kv_ids, active_count = _build_active_kv_ids(mask_2d, kv)
    akv_flat = active_kv_ids.reshape(-1).contiguous()
    acnt_flat = active_count.reshape(-1).contiguous()

    for bk in [16, 32, 64]:
        if bk > c_in or c_in % bk != 0:
            continue
        out_ck = implicit_gemm_crossk_forward(features, filters, ip, ipn, n_active, block_k=bk)
        if out_ck is None:
            print(f"  crossk BK{bk:3d}: COMPILE FAILED")
            continue
        torch.cuda.synchronize()
        err = (out_ck.float() - out_kpipe.float()).abs().max().item()

        ck_key = ("crossk", c_in, c_out, kv, "f32", bk)
        ck_launch, _ = CROSSK_COMPILED_KERNELS[ck_key]
        ck_out = torch.zeros(n_active, c_out, dtype=torch.float32, device=device)

        ck_us = bench_kernel(
            lambda l=ck_launch, o=ck_out: l(features_c, weights_packed, o, lut_flat, akv_flat, acnt_flat, num_tiles, n_active, stream)
        )
        delta = (ck_us / kpipe_us - 1) * 100
        print(f"  crossk BK{bk:3d}               {ck_us:8.1f} us  {delta:+6.1f}%  err={err:.1e}")


def main():
    print("=" * 78)
    print("Step 9: crossk sweep")
    print("=" * 78)
    for c_in, c_out, kv, n_active, density in CONFIGS:
        print(f"\n--- C_IN={c_in}  C_OUT={c_out}  KV={kv}  N={n_active}  density={density} ---")
        try:
            run_config(c_in, c_out, kv, n_active, density)
        except Exception as e:
            print(f"  ERROR: {e}")
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
