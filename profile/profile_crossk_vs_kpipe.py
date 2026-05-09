"""Head-to-head comparison: crossk (Step 8) vs kpipe baseline (Step 5).

Measures kernel launch time only (preprocessing is excluded).
"""
import argparse
import os
import sys
import time

import torch


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


def bench(label, fn, warmup=10, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    us = (time.perf_counter() - start) * 1e6 / max(iters, 1)
    print(f"  {label:40s}  {us:8.1f} us")
    return us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-active", type=int, default=20000)
    parser.add_argument("--c-in", type=int, default=64)
    parser.add_argument("--c-out", type=int, default=128)
    parser.add_argument("--kv", type=int, default=27)
    parser.add_argument("--density", type=float, default=0.3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from cumm.implicit_gemm_common import _get_hip_module, _pack_weights
    from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe import (
        MFMA_F32_16X16X4F32_N2_ASHARED_KPIPE_COMPILED_KERNELS,
        implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward,
    )
    from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_crossk import (
        CROSSK_COMPILED_KERNELS,
        _build_active_kv_ids,
        implicit_gemm_crossk_forward,
        implicit_gemm_crossk_prefetch_forward,
    )

    torch.manual_seed(0)
    device = "cuda"

    features = torch.randn(args.n_active, args.c_in, dtype=torch.float32, device=device) * 0.1
    filters = torch.randn(args.kv, args.c_in, args.c_out, dtype=torch.float32, device=device) * 0.1
    ip, ipn = make_subm_pairs(args.n_active, args.kv, args.density, device)
    block_m = 16
    num_tiles = (args.n_active + block_m - 1) // block_m

    print(f"\n=== N={args.n_active}  C_IN={args.c_in}  C_OUT={args.c_out}  KV={args.kv}  density={args.density} ===")

    # --- Shared preprocessing (done once) ---
    hip = _get_hip_module()
    assert hip is not None
    _, _, _, mask, _, _, inp_row_lut = hip.build_implicit_gemm_mask(ip, ipn, args.n_active, block_m)
    features_c = features.contiguous()
    weights_packed = _pack_weights(filters, 16)
    lut_flat = inp_row_lut.reshape(-1).contiguous()
    mask_flat = mask.reshape(-1).contiguous()
    stream = torch.cuda.current_stream()

    # --- Kpipe BK16 baseline (compile + cache hit) ---
    out_kpipe = implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward(
        features, filters, ip, ipn, args.n_active
    )
    assert out_kpipe is not None, "kpipe BK16 compile failed"
    torch.cuda.synchronize()

    kpipe_key = ("mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16", args.c_in, args.c_out, args.kv, "f32", 16, "direct")
    kpipe_launch, _ = MFMA_F32_16X16X4F32_N2_ASHARED_KPIPE_COMPILED_KERNELS[kpipe_key]
    out_buf = torch.zeros(args.n_active, args.c_out, dtype=torch.float32, device=device)

    def run_kpipe():
        kpipe_launch(features_c, weights_packed, out_buf, lut_flat, mask_flat, num_tiles, args.n_active, stream)

    kpipe_us = bench("kpipe BK16 (Step 5 baseline)", run_kpipe, warmup=args.warmup, iters=args.iters)

    # --- CrossK preprocessing ---
    mask_2d = mask.reshape(num_tiles, args.kv)
    active_kv_ids, active_count = _build_active_kv_ids(mask_2d, args.kv)
    akv_flat = active_kv_ids.reshape(-1).contiguous()
    acnt_flat = active_count.reshape(-1).contiguous()

    def bench_crossk(block_k, label):
        out_ck = implicit_gemm_crossk_forward(features, filters, ip, ipn, args.n_active, block_k=block_k)
        if out_ck is None:
            print(f"  {label}: compile failed")
            return None
        torch.cuda.synchronize()
        err = (out_ck.float() - out_kpipe.float()).abs().max().item()
        print(f"  {label} max error vs kpipe: {err:.6f}")

        ck_key = ("crossk", args.c_in, args.c_out, args.kv, "f32", block_k)
        ck_launch, _ = CROSSK_COMPILED_KERNELS[ck_key]
        ck_out = torch.zeros(args.n_active, args.c_out, dtype=torch.float32, device=device)

        def run_ck():
            ck_launch(features_c, weights_packed, ck_out, lut_flat, akv_flat, acnt_flat, num_tiles, args.n_active, stream)

        ck_us = bench(label, run_ck, warmup=args.warmup, iters=args.iters)
        print(f"  {label} vs kpipe BK16: {(ck_us/kpipe_us - 1)*100:+.1f}%")
        return ck_us

    bench_crossk(16, "crossk BK16")

    if args.c_in >= 32 and args.c_in % 32 == 0:
        bench_crossk(32, "crossk BK32")

    if args.c_in >= 64 and args.c_in % 64 == 0:
        bench_crossk(64, "crossk BK64")

    # --- CrossK Prefetch variants ---
    def bench_crossk_pf(block_k, label):
        out_pf = implicit_gemm_crossk_prefetch_forward(features, filters, ip, ipn, args.n_active, block_k=block_k)
        if out_pf is None:
            print(f"  {label}: compile failed")
            return None
        torch.cuda.synchronize()
        err = (out_pf.float() - out_kpipe.float()).abs().max().item()
        print(f"  {label} max error vs kpipe: {err:.6f}")

        pf_key = ("crossk_pf", args.c_in, args.c_out, args.kv, "f32", block_k)
        pf_launch, _ = CROSSK_COMPILED_KERNELS[pf_key]
        pf_out = torch.zeros(args.n_active, args.c_out, dtype=torch.float32, device=device)

        def run_pf():
            pf_launch(features_c, weights_packed, pf_out, lut_flat, akv_flat, acnt_flat, num_tiles, args.n_active, stream)

        pf_us = bench(label, run_pf, warmup=args.warmup, iters=args.iters)
        print(f"  {label} vs kpipe BK16: {(pf_us/kpipe_us - 1)*100:+.1f}%")
        return pf_us

    if args.c_in >= 32 and args.c_in % 32 == 0:
        bench_crossk_pf(32, "crossk-PF BK32")

    print()


if __name__ == "__main__":
    main()
