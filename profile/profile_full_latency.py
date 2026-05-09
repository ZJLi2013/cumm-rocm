"""Full latency benchmark: preprocessing + kernel, comparing dispatch path vs scalar_tile.

Usage:
    python profile/profile_full_latency.py [--iters 50]
"""
import argparse
import time
import torch

def bench_full(label, fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    us = (t1 - t0) / iters * 1e6
    print(f"  {label:40s} {us:10.1f} us")
    return us


def make_test_data(n_in, n_out, c_in, c_out, kv, density, device="cuda"):
    features = torch.randn(n_in, c_in, dtype=torch.float32, device=device) * 0.1
    filters = torch.randn(kv, c_in, c_out, dtype=torch.float32, device=device) * 0.1
    max_pairs = int(n_out * density)
    indice_pairs = torch.full((kv, 2, max_pairs), -1, dtype=torch.int32, device=device)
    indice_pair_num = torch.zeros(kv, dtype=torch.int32, device=device)
    for k in range(kv):
        nhot = max_pairs
        inp_idx = torch.randint(0, n_in, (nhot,), device=device, dtype=torch.int32)
        out_idx = torch.randperm(n_out, device=device, dtype=torch.int32)[:nhot]
        indice_pairs[k, 0, :nhot] = inp_idx
        indice_pairs[k, 1, :nhot] = out_idx
        indice_pair_num[k] = nhot
    return features, filters, indice_pairs, indice_pair_num, n_out


def run_config(n_in, n_out, c_in, c_out, kv, density, iters):
    print(f"\n=== N={n_in}, C_in={c_in}, C_out={c_out}, KV={kv}, density={density} ===")
    data = make_test_data(n_in, n_out, c_in, c_out, kv, density)
    features, filters, ip, ipn, nao = data

    from cumm.implicit_gemm_scalar_tile import implicit_gemm_scalar_tile_forward
    from cumm.implicit_gemm import implicit_gemm_forward, select_implicit_gemm_kernel

    desp = select_implicit_gemm_kernel(features.dtype, c_in, c_out)
    print(f"  Dispatch selects: {desp.name}")

    # scalar_tile full latency
    scalar_us = bench_full("scalar_tile (full)", lambda: implicit_gemm_scalar_tile_forward(
        features, filters, ip, ipn, nao
    ), iters=iters)

    # dispatch path full latency
    dispatch_us = bench_full(f"{desp.name} (full)", lambda: implicit_gemm_forward(
        features, filters, ip, ipn, nao
    ), iters=iters)

    delta = (dispatch_us / scalar_us - 1) * 100
    print(f"  {'dispatch vs scalar':40s} {delta:+.1f}%")
    return desp.name, scalar_us, dispatch_us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    configs = [
        (5000,  5000,  16,  32,  27, 0.3),
        (20000, 20000, 32,  32,  27, 0.3),
        (20000, 20000, 32,  64,  27, 0.3),
        (20000, 20000, 64,  128, 27, 0.3),
        (20000, 20000, 32,  32,   3, 0.5),
        (20000, 20000, 32,  32,   9, 0.3),
        (50000, 50000, 64,  128, 27, 0.3),
    ]

    print("Full Latency Benchmark (preprocessing + kernel)")
    print("=" * 70)
    results = []
    for n_in, n_out, c_in, c_out, kv, density in configs:
        name, s_us, d_us = run_config(n_in, n_out, c_in, c_out, kv, density, args.iters)
        results.append((f"{c_in}x{c_out} KV{kv}", name, s_us, d_us))

    print("\n\nSummary Table:")
    print(f"{'Config':20s} {'Kernel':35s} {'Scalar(us)':>12s} {'Dispatch(us)':>12s} {'Delta':>8s}")
    print("-" * 90)
    for config, name, s_us, d_us in results:
        delta = (d_us / s_us - 1) * 100
        print(f"{config:20s} {name:35s} {s_us:12.1f} {d_us:12.1f} {delta:+7.1f}%")


if __name__ == "__main__":
    main()
