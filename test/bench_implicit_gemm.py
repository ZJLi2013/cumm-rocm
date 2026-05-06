"""Benchmark: Output-tile-centric Implicit GEMM vs C++ for-loop indice_conv.

Compares:
  A) Python for-loop: 27x (index_select -> mm -> index_add_)
  B) Implicit GEMM V2: C++/HIP mask generation + FlyDSL fused kernel
     B.1) Full pipeline (mask gen + kernel)
     B.2) Mask generation only (C++/HIP sort + mask build)
     B.3) Kernel only (pre-computed mask)
"""
import time
import torch
import numpy as np


def make_subm_pairs(n_active, kv, density, device):
    """Simulate SubM conv indice pairs [kv, 2, N]."""
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
        inp = torch.randint(0, n_active, (nhot,), dtype=torch.int32, device=device)
        out = torch.randint(0, n_active, (nhot,), dtype=torch.int32, device=device)
        if k == kv // 2:
            out = inp.clone()
        indice_pairs[k, 0, :nhot] = inp
        indice_pairs[k, 1, :nhot] = out

    return indice_pairs, indice_pair_num


def reference_indice_conv(features, filters, indice_pairs, indice_pair_num, n_out):
    """Python for-loop: gather -> mm -> scatter per kv."""
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
    from cumm.implicit_gemm import implicit_gemm_forward, _get_hip_module

    features = torch.randn(n_active, c_in, dtype=dtype, device=device) * 0.1
    filters = torch.randn(kv, c_in, c_out, dtype=dtype, device=device) * 0.1
    ip, ipn = make_subm_pairs(n_active, kv, density, device)

    total_pairs = int(ipn.sum().item())
    print(f"\n{'='*75}")
    print(f"N={n_active}, C_in={c_in}, C_out={c_out}, kv={kv}, "
          f"density={density:.1%}, total_pairs={total_pairs}, dtype={dtype}")
    print(f"{'='*75}")

    # Correctness check
    ref = reference_indice_conv(features, filters, ip, ipn, n_active)
    impl = implicit_gemm_forward(features, filters, ip, ipn, n_active)
    if impl is not None:
        torch.cuda.synchronize()
        max_err = (impl.float() - ref.float()).abs().max().item()
        print(f"  Max error (implicit vs reference): {max_err:.6f}")
    else:
        print(f"  [WARN] Implicit GEMM failed, skipping")
        return

    # [A] Python for-loop baseline
    bench_fn(lambda: reference_indice_conv(features, filters, ip, ipn, n_active),
             label="[A] Python for-loop (gather+mm+scatter)")

    # [B] Full implicit GEMM pipeline (mask gen + kernel)
    bench_fn(lambda: implicit_gemm_forward(features, filters, ip, ipn, n_active),
             label="[B] Implicit GEMM V2 (full: mask + kernel)")

    # [B.2] Mask generation only
    hip = _get_hip_module()
    if hip is not None:
        BLOCK_M = 64
        bench_fn(lambda: hip.build_implicit_gemm_mask(ip, ipn, n_active, BLOCK_M),
                 label="[B.2] Mask generation only (C++/HIP sort+mask)")

    # [B.3] Kernel only (pre-computed mask)
    from cumm.implicit_gemm import _COMPILED_KERNELS
    dtype_str = 'f32' if dtype == torch.float32 else ('f16' if dtype == torch.float16 else 'bf16')
    key = (c_in, c_out, kv, dtype_str)
    if key in _COMPILED_KERNELS and hip is not None:
        launch_fn, block_m = _COMPILED_KERNELS[key]
        num_tiles = (n_active + block_m - 1) // block_m

        # Pre-compute mask
        sorted_inp, sorted_out, sorted_kv, mask, pair_start, pair_end = \
            hip.build_implicit_gemm_mask(ip, ipn, n_active, block_m)
        weights_flat = filters.reshape(-1).contiguous()
        features_c = features.contiguous()
        mask_flat = mask.reshape(-1).contiguous()
        ps_flat = pair_start.reshape(-1).contiguous()
        pe_flat = pair_end.reshape(-1).contiguous()

        def kernel_only():
            out_features = torch.zeros(n_active, c_out, dtype=torch.float32, device=device)
            stream = torch.cuda.current_stream()
            launch_fn(features_c, weights_flat, out_features,
                      sorted_inp, sorted_out,
                      mask_flat, ps_flat, pe_flat,
                      num_tiles, n_active, stream)
            return out_features

        bench_fn(kernel_only, label="[B.3] Kernel only (no mask gen)")


def main():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    configs = [
        # (n_active, c_in, c_out, kv, density, dtype)
        (5000,   16, 32,  27, 0.3, torch.float32),
        (5000,   16, 32,  27, 0.3, torch.float16),
        (20000,  32, 32,  27, 0.3, torch.float32),
        (20000,  32, 32,  27, 0.3, torch.float16),
        (50000,  32, 64,  27, 0.3, torch.float32),
        (20000,  64, 128, 27, 0.3, torch.float32),
        (20000,  32, 32,   1, 1.0, torch.float32),
    ]

    for cfg in configs:
        bench_config(*cfg, device=device)

    print("\n\nDone.")


if __name__ == "__main__":
    main()
