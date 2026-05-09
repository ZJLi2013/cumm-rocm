"""Kernel-only profiler driver for crossk-PF (prefetch) variant.

Compiles once, then repeatedly launches the cached crossk-PF kernel with
prebuilt preprocessing so profiler output focuses on the GPU kernel.
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["crossk", "crossk_pf"], default="crossk_pf")
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--n-active", type=int, default=20000)
    parser.add_argument("--c-in", type=int, default=64)
    parser.add_argument("--c-out", type=int, default=128)
    parser.add_argument("--kv", type=int, default=27)
    parser.add_argument("--density", type=float, default=0.3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from cumm.implicit_gemm_common import _get_hip_module, _pack_weights
    from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_crossk import (
        CROSSK_COMPILED_KERNELS,
        _build_active_kv_ids,
        implicit_gemm_crossk_forward,
        implicit_gemm_crossk_prefetch_forward,
    )

    torch.manual_seed(0)
    device = "cuda"
    block_m = 16

    features = torch.randn(args.n_active, args.c_in, dtype=torch.float32, device=device) * 0.1
    filters = torch.randn(args.kv, args.c_in, args.c_out, dtype=torch.float32, device=device) * 0.1
    ip, ipn = make_subm_pairs(args.n_active, args.kv, args.density, device)

    if args.kernel == "crossk_pf":
        compiled_out = implicit_gemm_crossk_prefetch_forward(
            features, filters, ip, ipn, args.n_active, block_k=args.block_k
        )
        cache_key = ("crossk_pf", args.c_in, args.c_out, args.kv, "f32", args.block_k)
    else:
        compiled_out = implicit_gemm_crossk_forward(
            features, filters, ip, ipn, args.n_active, block_k=args.block_k
        )
        cache_key = ("crossk", args.c_in, args.c_out, args.kv, "f32", args.block_k)

    if compiled_out is None:
        raise RuntimeError(f"{args.kernel} BK{args.block_k} failed to compile")
    torch.cuda.synchronize()

    launch_fn, _ = CROSSK_COMPILED_KERNELS[cache_key]

    hip = _get_hip_module()
    assert hip is not None
    _, _, _, mask, _, _, inp_row_lut = hip.build_implicit_gemm_mask(
        ip, ipn, args.n_active, block_m
    )
    num_tiles = (args.n_active + block_m - 1) // block_m
    features_c = features.contiguous()
    weights_packed = _pack_weights(filters, 16)
    lut_flat = inp_row_lut.reshape(-1).contiguous()
    mask_2d = mask.reshape(num_tiles, args.kv)
    active_kv_ids, active_count = _build_active_kv_ids(mask_2d, args.kv)
    akv_flat = active_kv_ids.reshape(-1).contiguous()
    acnt_flat = active_count.reshape(-1).contiguous()
    out_features = torch.empty(args.n_active, args.c_out, dtype=torch.float32, device=device)
    stream = torch.cuda.current_stream()

    def run_kernel():
        launch_fn(
            features_c, weights_packed, out_features,
            lut_flat, akv_flat, acnt_flat,
            num_tiles, args.n_active, stream,
        )

    for _ in range(args.warmup):
        run_kernel()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.iters):
        run_kernel()
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) * 1e6 / max(args.iters, 1)

    print(
        f"profile_{args.kernel}_kernel",
        f"BK={args.block_k}",
        f"shape=N{args.n_active}_C{args.c_in}_{args.c_out}_KV{args.kv}_D{args.density}",
        f"iters={args.iters}",
        f"avg_us={elapsed_us:.3f}",
    )


if __name__ == "__main__":
    main()
