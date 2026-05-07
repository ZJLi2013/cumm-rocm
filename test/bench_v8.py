"""Benchmark: V8a vs V8c implicit GEMM."""
import time

import numpy as np
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
    out = torch.zeros(n_out, c_out, dtype=features.dtype, device=features.device)
    for k in range(kv):
        nhot = int(indice_pair_num[k].item())
        if nhot == 0:
            continue
        inp_ids = indice_pairs[k, 0, :nhot].long()
        out_ids = indice_pairs[k, 1, :nhot].long()
        out.index_add_(0, out_ids, features[inp_ids] @ filters[k])
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
        times.append((time.perf_counter() - t0) * 1e6)
    arr = np.array(times)
    print(
        f"  {label:<55} avg={arr.mean():8.0f}us  "
        f"med={np.median(arr):8.0f}us  min={arr.min():8.0f}us"
    )
    return arr.mean(), np.median(arr), arr.min()


def _dtype_str(dtype):
    return "f32" if dtype == torch.float32 else ("f16" if dtype == torch.float16 else "bf16")


def bench_config(n_active, c_in, c_out, kv, density, dtype, device):
    from cumm.implicit_gemm import (
        _V8A_COMPILED_KERNELS,
        _V8C_COMPILED_KERNELS,
        _get_hip_module,
        implicit_gemm_v8a_forward,
        implicit_gemm_v8c_forward,
    )
    from cumm.implicit_gemm_v6 import _pack_weights

    features = torch.randn(n_active, c_in, dtype=dtype, device=device) * 0.1
    filters = torch.randn(kv, c_in, c_out, dtype=dtype, device=device) * 0.1
    ip, ipn = make_subm_pairs(n_active, kv, density, device)

    total_pairs = int(ipn.sum().item())
    print(f"\n{'=' * 75}")
    print(
        f"N={n_active}, C_in={c_in}, C_out={c_out}, kv={kv}, "
        f"density={density:.1%}, total_pairs={total_pairs}, dtype={dtype}"
    )
    print(f"{'=' * 75}")

    ref = reference_indice_conv(features, filters, ip, ipn, n_active)
    bench_fn(
        lambda: reference_indice_conv(features, filters, ip, ipn, n_active),
        label="[A] Python for-loop (gather+mm+scatter)",
    )

    hip = _get_hip_module()
    dtype_str = _dtype_str(dtype)
    weights_packed_v8a = _pack_weights(filters, min(64, c_out))
    weights_packed_v8c = _pack_weights(filters, 16)
    features_c = features.contiguous()

    # V8a baseline
    impl_v8a = implicit_gemm_v8a_forward(features, filters, ip, ipn, n_active)
    if impl_v8a is not None:
        print(f"  V8a max error: {(impl_v8a.float() - ref.float()).abs().max().item():.6f}")
        bench_fn(
            lambda: implicit_gemm_v8a_forward(features, filters, ip, ipn, n_active),
            label="[I] V8a tile-owned scalar full",
        )
        v8a_key = ("v8a", c_in, c_out, kv, dtype_str)
        if v8a_key in _V8A_COMPILED_KERNELS and hip is not None:
            launch_fn, block_m = _V8A_COMPILED_KERNELS[v8a_key]
            num_tiles = (n_active + block_m - 1) // block_m
            _, _, _, mask, _, _, lut = hip.build_implicit_gemm_mask(ip, ipn, n_active, block_m)
            mask_flat = mask.reshape(-1).contiguous()
            lut_flat = lut.reshape(-1).contiguous()

            def v8a_kernel_only():
                out_features = torch.zeros(n_active, c_out, dtype=torch.float32, device=device)
                launch_fn(
                    features_c,
                    weights_packed_v8a,
                    out_features,
                    lut_flat,
                    mask_flat,
                    num_tiles,
                    n_active,
                    torch.cuda.current_stream(),
                )
                return out_features

            bench_fn(v8a_kernel_only, label="[I.3] V8a Kernel only")
    else:
        print("  [WARN] V8a failed, skipping")

    # V8c
    impl_v8c = implicit_gemm_v8c_forward(features, filters, ip, ipn, n_active)
    if impl_v8c is not None:
        print(f"  V8c max error: {(impl_v8c.float() - ref.float()).abs().max().item():.6f}")
        bench_fn(
            lambda: implicit_gemm_v8c_forward(features, filters, ip, ipn, n_active),
            label="[J] V8c minimal MFMA full",
        )
        v8c_key = ("v8c", c_in, c_out, kv, dtype_str)
        if v8c_key in _V8C_COMPILED_KERNELS and hip is not None:
            launch_fn, block_m = _V8C_COMPILED_KERNELS[v8c_key]
            num_tiles = (n_active + block_m - 1) // block_m
            _, _, _, mask, _, _, lut = hip.build_implicit_gemm_mask(ip, ipn, n_active, block_m)
            mask_flat = mask.reshape(-1).contiguous()
            lut_flat = lut.reshape(-1).contiguous()

            def v8c_kernel_only():
                out_features = torch.zeros(n_active, c_out, dtype=torch.float32, device=device)
                launch_fn(
                    features_c,
                    weights_packed_v8c,
                    out_features,
                    lut_flat,
                    mask_flat,
                    num_tiles,
                    n_active,
                    torch.cuda.current_stream(),
                )
                return out_features

            bench_fn(v8c_kernel_only, label="[J.3] V8c Kernel only")
    else:
        print("  [WARN] V8c failed, skipping")

    if hip is not None:
        bench_fn(
            lambda: hip.build_implicit_gemm_mask(ip, ipn, n_active, 64),
            label="[F] Mask+LUT generation only",
        )


def main():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    configs = [
        (5000, 16, 32, 27, 0.3, torch.float32),
        (20000, 32, 32, 27, 0.3, torch.float32),
        (20000, 64, 128, 27, 0.3, torch.float32),
    ]
    for cfg in configs:
        bench_config(*cfg, device=device)
    print("\n\nDone.")


if __name__ == "__main__":
    main()
