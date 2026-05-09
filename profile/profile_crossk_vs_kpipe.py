"""Head-to-head comparison: crossk (Step 8) vs kpipe baseline (Step 5)."""
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

    from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe import (
        implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward,
    )
    from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_crossk import (
        implicit_gemm_crossk_forward,
    )

    torch.manual_seed(0)
    device = "cuda"

    features = torch.randn(args.n_active, args.c_in, dtype=torch.float32, device=device) * 0.1
    filters = torch.randn(args.kv, args.c_in, args.c_out, dtype=torch.float32, device=device) * 0.1
    ip, ipn = make_subm_pairs(args.n_active, args.kv, args.density, device)

    print(f"\n=== N={args.n_active}  C_IN={args.c_in}  C_OUT={args.c_out}  KV={args.kv}  density={args.density} ===")

    # Kpipe BK16 baseline
    out_kpipe = implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward(
        features, filters, ip, ipn, args.n_active
    )
    assert out_kpipe is not None, "kpipe BK16 compile failed"
    torch.cuda.synchronize()

    kpipe_us = bench(
        "kpipe BK16 (Step 5 baseline)",
        lambda: implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward(
            features, filters, ip, ipn, args.n_active
        ),
        warmup=args.warmup,
        iters=args.iters,
    )

    # CrossK BK16
    out_ck16 = implicit_gemm_crossk_forward(features, filters, ip, ipn, args.n_active, block_k=16)
    if out_ck16 is not None:
        torch.cuda.synchronize()
        err16 = (out_ck16.float() - out_kpipe.float()).abs().max().item()
        print(f"  crossk BK16 max error vs kpipe: {err16:.6f}")
        ck16_us = bench(
            "crossk BK16",
            lambda: implicit_gemm_crossk_forward(features, filters, ip, ipn, args.n_active, block_k=16),
            warmup=args.warmup,
            iters=args.iters,
        )
        print(f"  crossk BK16 vs kpipe BK16: {(ck16_us/kpipe_us - 1)*100:+.1f}%")
    else:
        print("  crossk BK16: skipped (c_in constraint)")

    # CrossK BK32
    if args.c_in >= 32 and args.c_in % 32 == 0:
        out_ck32 = implicit_gemm_crossk_forward(features, filters, ip, ipn, args.n_active, block_k=32)
        if out_ck32 is not None:
            torch.cuda.synchronize()
            err32 = (out_ck32.float() - out_kpipe.float()).abs().max().item()
            print(f"  crossk BK32 max error vs kpipe: {err32:.6f}")
            ck32_us = bench(
                "crossk BK32",
                lambda: implicit_gemm_crossk_forward(features, filters, ip, ipn, args.n_active, block_k=32),
                warmup=args.warmup,
                iters=args.iters,
            )
            print(f"  crossk BK32 vs kpipe BK16: {(ck32_us/kpipe_us - 1)*100:+.1f}%")
    else:
        print("  crossk BK32: skipped (c_in < 32 or not divisible)")

    # CrossK BK64
    if args.c_in >= 64 and args.c_in % 64 == 0:
        out_ck64 = implicit_gemm_crossk_forward(features, filters, ip, ipn, args.n_active, block_k=64)
        if out_ck64 is not None:
            torch.cuda.synchronize()
            err64 = (out_ck64.float() - out_kpipe.float()).abs().max().item()
            print(f"  crossk BK64 max error vs kpipe: {err64:.6f}")
            ck64_us = bench(
                "crossk BK64",
                lambda: implicit_gemm_crossk_forward(features, filters, ip, ipn, args.n_active, block_k=64),
                warmup=args.warmup,
                iters=args.iters,
            )
            print(f"  crossk BK64 vs kpipe BK16: {(ck64_us/kpipe_us - 1)*100:+.1f}%")
    else:
        print("  crossk BK64: skipped (c_in < 64 or not divisible)")

    print()


if __name__ == "__main__":
    main()
