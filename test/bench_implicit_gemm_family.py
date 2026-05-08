"""Benchmark current implicit GEMM kernel family."""
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
        MFMA_F32_16X16X4F32_COMPILED_KERNELS,
        MFMA_F32_16X16X4F32_N2_COMPILED_KERNELS,
        MFMA_F32_16X16X4F32_N2_ASHARED_COMPILED_KERNELS,
        MFMA_F32_16X16X4F32_N2_ASHARED_KPIPE_COMPILED_KERNELS,
        MFMA_F32_32X32X2F32_COMPILED_KERNELS,
        SCALAR_TILE_COMPILED_KERNELS,
        _get_hip_module,
        implicit_gemm_forward,
        implicit_gemm_mfma_f32_16x16x4f32_forward,
        implicit_gemm_mfma_f32_16x16x4f32_n2_forward,
        implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_forward,
        implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward,
        implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk32_forward,
        implicit_gemm_mfma_f32_32x32x2f32_forward,
        implicit_gemm_scalar_tile_forward,
        select_implicit_gemm_kernel,
    )
    from cumm.implicit_gemm_common import _pack_weights

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

    selected = select_implicit_gemm_kernel(dtype, c_in, c_out)
    print(f"  dispatch selected: {selected.name}")
    impl_dispatch = implicit_gemm_forward(features, filters, ip, ipn, n_active)
    if impl_dispatch is not None:
        print(
            f"  dispatch max error: "
            f"{(impl_dispatch.float() - ref.float()).abs().max().item():.6f}"
        )
        bench_fn(
            lambda: implicit_gemm_forward(features, filters, ip, ipn, n_active),
            label="[D] dispatch full",
        )
    else:
        print("  [WARN] dispatch failed, continuing per-kernel benchmarks")

    hip = _get_hip_module()
    dtype_str = _dtype_str(dtype)
    weights_packed_scalar = _pack_weights(filters, min(64, c_out))
    weights_packed_mfma = _pack_weights(filters, 16)
    weights_packed_mfma32 = _pack_weights(filters, 32)
    features_c = features.contiguous()

    # scalar_tile fallback
    impl_scalar = implicit_gemm_scalar_tile_forward(features, filters, ip, ipn, n_active)
    if impl_scalar is not None:
        print(f"  scalar_tile max error: {(impl_scalar.float() - ref.float()).abs().max().item():.6f}")
        bench_fn(
            lambda: implicit_gemm_scalar_tile_forward(features, filters, ip, ipn, n_active),
            label="[I] scalar_tile full",
        )
        scalar_key = ("scalar_tile", c_in, c_out, kv, dtype_str)
        if scalar_key in SCALAR_TILE_COMPILED_KERNELS and hip is not None:
            launch_fn, block_m = SCALAR_TILE_COMPILED_KERNELS[scalar_key]
            num_tiles = (n_active + block_m - 1) // block_m
            _, _, _, mask, _, _, lut = hip.build_implicit_gemm_mask(ip, ipn, n_active, block_m)
            mask_flat = mask.reshape(-1).contiguous()
            lut_flat = lut.reshape(-1).contiguous()

            def scalar_kernel_only():
                out_features = torch.zeros(n_active, c_out, dtype=torch.float32, device=device)
                launch_fn(
                    features_c,
                    weights_packed_scalar,
                    out_features,
                    lut_flat,
                    mask_flat,
                    num_tiles,
                    n_active,
                    torch.cuda.current_stream(),
                )
                return out_features

            bench_fn(scalar_kernel_only, label="[I.3] scalar_tile kernel only")
    else:
        print("  [WARN] scalar_tile failed, skipping")

    # mfma_f32_16x16x4f32
    impl_mfma = implicit_gemm_mfma_f32_16x16x4f32_forward(features, filters, ip, ipn, n_active)
    if impl_mfma is not None:
        print(f"  mfma_f32_16x16x4f32 max error: {(impl_mfma.float() - ref.float()).abs().max().item():.6f}")
        bench_fn(
            lambda: implicit_gemm_mfma_f32_16x16x4f32_forward(features, filters, ip, ipn, n_active),
            label="[J] mfma_f32_16x16x4f32 full",
        )
        mfma_key = ("mfma_f32_16x16x4f32", c_in, c_out, kv, dtype_str)
        if mfma_key in MFMA_F32_16X16X4F32_COMPILED_KERNELS and hip is not None:
            launch_fn, block_m = MFMA_F32_16X16X4F32_COMPILED_KERNELS[mfma_key]
            num_tiles = (n_active + block_m - 1) // block_m
            _, _, _, mask, _, _, lut = hip.build_implicit_gemm_mask(ip, ipn, n_active, block_m)
            mask_flat = mask.reshape(-1).contiguous()
            lut_flat = lut.reshape(-1).contiguous()

            def mfma_kernel_only():
                out_features = torch.zeros(n_active, c_out, dtype=torch.float32, device=device)
                launch_fn(
                    features_c,
                    weights_packed_mfma,
                    out_features,
                    lut_flat,
                    mask_flat,
                    num_tiles,
                    n_active,
                    torch.cuda.current_stream(),
                )
                return out_features

            bench_fn(mfma_kernel_only, label="[J.3] mfma_f32_16x16x4f32 kernel only")
    else:
        print("  [WARN] mfma_f32_16x16x4f32 failed, skipping")

    # mfma_f32_16x16x4f32_n2
    impl_mfma_n2 = implicit_gemm_mfma_f32_16x16x4f32_n2_forward(
        features, filters, ip, ipn, n_active
    )
    if impl_mfma_n2 is not None:
        print(
            "  mfma_f32_16x16x4f32_n2 max error: "
            f"{(impl_mfma_n2.float() - ref.float()).abs().max().item():.6f}"
        )
        bench_fn(
            lambda: implicit_gemm_mfma_f32_16x16x4f32_n2_forward(
                features, filters, ip, ipn, n_active
            ),
            label="[K] mfma_f32_16x16x4f32_n2 full",
        )
        mfma_n2_key = ("mfma_f32_16x16x4f32_n2", c_in, c_out, kv, dtype_str)
        if mfma_n2_key in MFMA_F32_16X16X4F32_N2_COMPILED_KERNELS and hip is not None:
            launch_fn, block_m = MFMA_F32_16X16X4F32_N2_COMPILED_KERNELS[mfma_n2_key]
            num_tiles = (n_active + block_m - 1) // block_m
            _, _, _, mask, _, _, lut = hip.build_implicit_gemm_mask(ip, ipn, n_active, block_m)
            mask_flat = mask.reshape(-1).contiguous()
            lut_flat = lut.reshape(-1).contiguous()

            def mfma_n2_kernel_only():
                out_features = torch.zeros(n_active, c_out, dtype=torch.float32, device=device)
                launch_fn(
                    features_c,
                    weights_packed_mfma,
                    out_features,
                    lut_flat,
                    mask_flat,
                    num_tiles,
                    n_active,
                    torch.cuda.current_stream(),
                )
                return out_features

            bench_fn(
                mfma_n2_kernel_only,
                label="[K.3] mfma_f32_16x16x4f32_n2 kernel only",
            )
    else:
        print("  [WARN] mfma_f32_16x16x4f32_n2 failed, skipping")

    # mfma_f32_16x16x4f32_n2_ashared
    impl_mfma_n2_ashared = implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_forward(
        features, filters, ip, ipn, n_active
    )
    if impl_mfma_n2_ashared is not None:
        print(
            "  mfma_f32_16x16x4f32_n2_ashared max error: "
            f"{(impl_mfma_n2_ashared.float() - ref.float()).abs().max().item():.6f}"
        )
        bench_fn(
            lambda: implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_forward(
                features, filters, ip, ipn, n_active
            ),
            label="[K2] mfma_f32_16x16x4f32_n2_ashared full",
        )
        mfma_n2_ashared_key = (
            "mfma_f32_16x16x4f32_n2_ashared",
            c_in,
            c_out,
            kv,
            dtype_str,
        )
        if (
            mfma_n2_ashared_key in MFMA_F32_16X16X4F32_N2_ASHARED_COMPILED_KERNELS
            and hip is not None
        ):
            launch_fn, block_m = MFMA_F32_16X16X4F32_N2_ASHARED_COMPILED_KERNELS[
                mfma_n2_ashared_key
            ]
            num_tiles = (n_active + block_m - 1) // block_m
            _, _, _, mask, _, _, lut = hip.build_implicit_gemm_mask(ip, ipn, n_active, block_m)
            mask_flat = mask.reshape(-1).contiguous()
            lut_flat = lut.reshape(-1).contiguous()

            def mfma_n2_ashared_kernel_only():
                out_features = torch.zeros(n_active, c_out, dtype=torch.float32, device=device)
                launch_fn(
                    features_c,
                    weights_packed_mfma,
                    out_features,
                    lut_flat,
                    mask_flat,
                    num_tiles,
                    n_active,
                    torch.cuda.current_stream(),
                )
                return out_features

            bench_fn(
                mfma_n2_ashared_kernel_only,
                label="[K2.3] mfma_f32_16x16x4f32_n2_ashared kernel only",
            )
    else:
        print("  [WARN] mfma_f32_16x16x4f32_n2_ashared failed, skipping")

    for block_k, kpipe_forward in (
        (16, implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward),
        (32, implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk32_forward),
    ):
        kpipe_name = f"mfma_f32_16x16x4f32_n2_ashared_kpipe_bk{block_k}"
        impl_kpipe = kpipe_forward(features, filters, ip, ipn, n_active)
        if impl_kpipe is not None:
            print(
                f"  {kpipe_name} max error: "
                f"{(impl_kpipe.float() - ref.float()).abs().max().item():.6f}"
            )
            bench_fn(
                lambda fn=kpipe_forward: fn(features, filters, ip, ipn, n_active),
                label=f"[K3-bk{block_k}] {kpipe_name} full",
            )
            kpipe_key = (kpipe_name, c_in, c_out, kv, dtype_str, block_k)
            if (
                kpipe_key in MFMA_F32_16X16X4F32_N2_ASHARED_KPIPE_COMPILED_KERNELS
                and hip is not None
            ):
                launch_fn, block_m = (
                    MFMA_F32_16X16X4F32_N2_ASHARED_KPIPE_COMPILED_KERNELS[kpipe_key]
                )
                num_tiles = (n_active + block_m - 1) // block_m
                _, _, _, mask, _, _, lut = hip.build_implicit_gemm_mask(
                    ip, ipn, n_active, block_m
                )
                mask_flat = mask.reshape(-1).contiguous()
                lut_flat = lut.reshape(-1).contiguous()

                def mfma_n2_ashared_kpipe_kernel_only():
                    out_features = torch.zeros(
                        n_active, c_out, dtype=torch.float32, device=device
                    )
                    launch_fn(
                        features_c,
                        weights_packed_mfma,
                        out_features,
                        lut_flat,
                        mask_flat,
                        num_tiles,
                        n_active,
                        torch.cuda.current_stream(),
                    )
                    return out_features

                bench_fn(
                    mfma_n2_ashared_kpipe_kernel_only,
                    label=f"[K3-bk{block_k}.3] {kpipe_name} kernel only",
                )
        else:
            print(f"  [WARN] {kpipe_name} failed, skipping")

    # mfma_f32_32x32x2f32
    impl_mfma32 = implicit_gemm_mfma_f32_32x32x2f32_forward(
        features, filters, ip, ipn, n_active
    )
    if impl_mfma32 is not None:
        print(
            "  mfma_f32_32x32x2f32 max error: "
            f"{(impl_mfma32.float() - ref.float()).abs().max().item():.6f}"
        )
        bench_fn(
            lambda: implicit_gemm_mfma_f32_32x32x2f32_forward(
                features, filters, ip, ipn, n_active
            ),
            label="[L] mfma_f32_32x32x2f32 full",
        )
        mfma32_key = ("mfma_f32_32x32x2f32", c_in, c_out, kv, dtype_str)
        if mfma32_key in MFMA_F32_32X32X2F32_COMPILED_KERNELS and hip is not None:
            launch_fn, block_m = MFMA_F32_32X32X2F32_COMPILED_KERNELS[mfma32_key]
            num_tiles = (n_active + block_m - 1) // block_m
            _, _, _, mask, _, _, lut = hip.build_implicit_gemm_mask(ip, ipn, n_active, block_m)
            mask_flat = mask.reshape(-1).contiguous()
            lut_flat = lut.reshape(-1).contiguous()

            def mfma32_kernel_only():
                out_features = torch.zeros(n_active, c_out, dtype=torch.float32, device=device)
                launch_fn(
                    features_c,
                    weights_packed_mfma32,
                    out_features,
                    lut_flat,
                    mask_flat,
                    num_tiles,
                    n_active,
                    torch.cuda.current_stream(),
                )
                return out_features

            bench_fn(
                mfma32_kernel_only,
                label="[L.3] mfma_f32_32x32x2f32 kernel only",
            )
    else:
        print("  [WARN] mfma_f32_32x32x2f32 failed, skipping")

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
