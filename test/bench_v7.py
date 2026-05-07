"""Benchmark: V4 vs V6 vs V7 implicit GEMM (skip V3/V5)."""
import time
import torch
import numpy as np


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

    for k in range(kv):
        nhot = nhot_per_kv[k]
        indice_pair_num[k] = nhot
        inp = torch.randperm(n_active, device=device, dtype=torch.int32)[:nhot]
        out = torch.randperm(n_active, device=device, dtype=torch.int32)[:nhot]
        if k == kv // 2:
            out = inp.clone()
        indice_pairs[k, 0, :nhot] = inp
        indice_pairs[k, 1, :nhot] = out

    return indice_pairs, indice_pair_num


def reference_indice_conv(features, filters, indice_pairs, indice_pair_num, n_out):
    kv = filters.shape[0]
    c_out = filters.shape[2]
    device = features.device
    dtype = features.dtype
    out = torch.zeros(n_out, c_out, dtype=dtype, device=device)
    for k in range(kv):
        nhot = int(indice_pair_num[k].item())
        if nhot == 0:
            continue
        inp_ids = indice_pairs[k, 0, :nhot].long()
        out_ids = indice_pairs[k, 1, :nhot].long()
        gathered = features[inp_ids]
        result = gathered @ filters[k]
        out.index_add_(0, out_ids, result)
    return out


def bench_fn(fn, warmup=10, repeats=50, label=""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) * 1e6
        times.append(t)
    arr = np.array(times)
    avg = arr.mean()
    med = np.median(arr)
    mi = arr.min()
    print(f"  {label:<55} avg={avg:8.0f}us  med={med:8.0f}us  min={mi:8.0f}us")
    return avg, med, mi


def bench_config(n_active, c_in, c_out, kv, density, dtype, device):
    from cumm.implicit_gemm import (
        implicit_gemm_v4_forward, implicit_gemm_v6_forward, implicit_gemm_v7_forward,
        _get_hip_module, _V4_COMPILED_KERNELS, _V6_COMPILED_KERNELS, _V7_COMPILED_KERNELS,
    )

    features = torch.randn(n_active, c_in, dtype=dtype, device=device) * 0.1
    filters = torch.randn(kv, c_in, c_out, dtype=dtype, device=device) * 0.1
    ip, ipn = make_subm_pairs(n_active, kv, density, device)

    total_pairs = int(ipn.sum().item())
    print(f"\n{'='*75}")
    print(f"N={n_active}, C_in={c_in}, C_out={c_out}, kv={kv}, "
          f"density={density:.1%}, total_pairs={total_pairs}, dtype={dtype}")
    print(f"{'='*75}")

    ref = reference_indice_conv(features, filters, ip, ipn, n_active)
    bench_fn(lambda: reference_indice_conv(features, filters, ip, ipn, n_active),
             label="[A] Python for-loop (gather+mm+scatter)")

    hip = _get_hip_module()

    # V4 baseline
    impl_v4 = implicit_gemm_v4_forward(features, filters, ip, ipn, n_active)
    if impl_v4 is not None:
        max_err = (impl_v4.float() - ref.float()).abs().max().item()
        print(f"  V4 max error: {max_err:.6f}")
        bench_fn(lambda: implicit_gemm_v4_forward(features, filters, ip, ipn, n_active),
                 label="[C] Implicit GEMM V4 (full)")

        dtype_str = 'f32' if dtype == torch.float32 else ('f16' if dtype == torch.float16 else 'bf16')
        v4_key = ('v4', c_in, c_out, kv, dtype_str)
        if v4_key in _V4_COMPILED_KERNELS and hip is not None:
            launch_fn, block_m = _V4_COMPILED_KERNELS[v4_key]
            num_tiles = (n_active + block_m - 1) // block_m
            _, _, _, mask, _, _, lut = hip.build_implicit_gemm_mask(ip, ipn, n_active, block_m)
            weights_flat = filters.reshape(-1).contiguous()
            features_c = features.contiguous()
            mask_flat = mask.reshape(-1).contiguous()
            lut_flat = lut.reshape(-1).contiguous()

            def v4_kernel_only():
                out_features = torch.zeros(n_active, c_out, dtype=torch.float32, device=device)
                launch_fn(features_c, weights_flat, out_features,
                          lut_flat, mask_flat,
                          num_tiles, n_active, torch.cuda.current_stream())
                return out_features

            bench_fn(v4_kernel_only, label="[C.3] V4 Kernel only")

    # V6
    impl_v6 = implicit_gemm_v6_forward(features, filters, ip, ipn, n_active)
    if impl_v6 is not None:
        max_err = (impl_v6.float() - ref.float()).abs().max().item()
        print(f"  V6 max error: {max_err:.6f}")

        dtype_str = 'f32' if dtype == torch.float32 else ('f16' if dtype == torch.float16 else 'bf16')
        v6_key = ('v6', c_in, c_out, kv, dtype_str)
        if v6_key in _V6_COMPILED_KERNELS and hip is not None:
            launch_fn, block_m = _V6_COMPILED_KERNELS[v6_key]
            num_tiles = (n_active + block_m - 1) // block_m
            _, _, _, mask, _, _, lut = hip.build_implicit_gemm_mask(ip, ipn, n_active, block_m)
            from cumm.implicit_gemm_v6 import _pack_weights, C_OUT_TILE_MAX
            c_out_tile = min(C_OUT_TILE_MAX, c_out)
            weights_packed = _pack_weights(filters, c_out_tile)
            features_c = features.contiguous()
            mask_flat = mask.reshape(-1).contiguous()
            lut_flat = lut.reshape(-1).contiguous()

            def v6_kernel_only():
                out_features = torch.zeros(n_active, c_out, dtype=torch.float32, device=device)
                launch_fn(features_c, weights_packed, out_features,
                          lut_flat, mask_flat,
                          num_tiles, n_active, torch.cuda.current_stream())
                return out_features

            bench_fn(v6_kernel_only, label="[G.3] V6 Kernel only")

    # V7
    impl_v7 = implicit_gemm_v7_forward(features, filters, ip, ipn, n_active)
    if impl_v7 is not None:
        max_err = (impl_v7.float() - ref.float()).abs().max().item()
        print(f"  V7 max error: {max_err:.6f}")
        bench_fn(lambda: implicit_gemm_v7_forward(features, filters, ip, ipn, n_active),
                 label="[H] Implicit GEMM V7 double-buf (full)")

        dtype_str = 'f32' if dtype == torch.float32 else ('f16' if dtype == torch.float16 else 'bf16')
        v7_key = ('v7', c_in, c_out, kv, dtype_str)
        if v7_key in _V7_COMPILED_KERNELS and hip is not None:
            launch_fn, block_m = _V7_COMPILED_KERNELS[v7_key]
            num_tiles = (n_active + block_m - 1) // block_m
            _, _, _, mask, _, _, lut = hip.build_implicit_gemm_mask(ip, ipn, n_active, block_m)
            from cumm.implicit_gemm_v6 import _pack_weights, C_OUT_TILE_MAX
            c_out_tile = min(C_OUT_TILE_MAX, c_out)
            weights_packed_v7 = _pack_weights(filters, c_out_tile)
            features_c = features.contiguous()
            mask_flat = mask.reshape(-1).contiguous()
            lut_flat = lut.reshape(-1).contiguous()

            def v7_kernel_only():
                out_features = torch.zeros(n_active, c_out, dtype=torch.float32, device=device)
                launch_fn(features_c, weights_packed_v7, out_features,
                          lut_flat, mask_flat,
                          num_tiles, n_active, torch.cuda.current_stream())
                return out_features

            bench_fn(v7_kernel_only, label="[H.3] V7 Kernel only (double-buf)")
    else:
        print(f"  [WARN] V7 failed, skipping")

    if hip is not None:
        BLOCK_M = 64
        bench_fn(lambda: hip.build_implicit_gemm_mask(ip, ipn, n_active, BLOCK_M),
                 label="[F] Mask+LUT generation only")


def main():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    configs = [
        (5000,   16, 32,  27, 0.3, torch.float32),
        (20000,  32, 32,  27, 0.3, torch.float32),
        (20000,  64, 128, 27, 0.3, torch.float32),
    ]

    for cfg in configs:
        bench_config(*cfg, device=device)

    print("\n\nDone.")


if __name__ == "__main__":
    main()
