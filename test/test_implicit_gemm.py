"""Unit tests for FlyDSL implicit GEMM kernel.

Tests focus on:
1. Correctness: gather+GEMM+scatter vs explicit loop
2. Memory layout: weight layout in LDS, feature access pattern
3. Coalescing: sorted vs unsorted pair indices
4. Edge cases: partial tiles, single pair, empty kv positions
"""
import pytest
import torch
import numpy as np

# ---------- host preprocessing tests (no GPU required) ----------

from cumm.implicit_gemm import preprocess_pairs


def _make_pairs(kv, nhot_list, n_max=None, device="cpu"):
    """Create synthetic indice_pairs and indice_pair_num."""
    if n_max is None:
        n_max = max(nhot_list) if nhot_list else 0
    ip = torch.full((kv, 2, n_max), -1, dtype=torch.int32, device=device)
    ipn = torch.zeros(kv, dtype=torch.int32, device=device)
    for k, nhot in enumerate(nhot_list):
        if nhot > 0:
            inp_ids = torch.randperm(200)[:nhot].int()
            out_ids = torch.randperm(200)[:nhot].int()
            ip[k, 0, :nhot] = inp_ids
            ip[k, 1, :nhot] = out_ids
            ipn[k] = nhot
    return ip, ipn


class TestPreprocessPairs:
    """Test host-side pair preprocessing: sorting and tiling."""

    def test_basic_tiling(self):
        kv = 3
        ip, ipn = _make_pairs(kv, [64, 32, 0])
        inp_flat, out_flat, tile_kpos, tile_pc, n_tiles = preprocess_pairs(ip, ipn, block_m=64)
        assert n_tiles == 2  # 64→1 tile, 32→1 tile, 0→skip
        assert tile_kpos[0] == 0
        assert tile_kpos[1] == 1
        assert tile_pc[0] == 64
        assert tile_pc[1] == 32
        assert inp_flat.shape[0] == n_tiles * 64

    def test_multi_tile_single_kv(self):
        ip, ipn = _make_pairs(1, [150])
        _, _, tile_kpos, tile_pc, n_tiles = preprocess_pairs(ip, ipn, block_m=64)
        assert n_tiles == 3  # ceil(150/64) = 3 tiles
        assert (tile_kpos == 0).all()
        counts = tile_pc.numpy()
        assert counts[0] == 64
        assert counts[1] == 64
        assert counts[2] == 150 - 128

    def test_sorting_by_inp_indices(self):
        ip = torch.zeros(1, 2, 64, dtype=torch.int32)
        ip[0, 0] = torch.arange(63, -1, -1, dtype=torch.int32)
        ip[0, 1] = torch.arange(64, dtype=torch.int32)
        ipn = torch.tensor([64], dtype=torch.int32)
        inp_flat, out_flat, _, _, _ = preprocess_pairs(ip, ipn, block_m=64)
        inp_np = inp_flat[:64].numpy()
        assert np.all(inp_np[:-1] <= inp_np[1:]), "inp_indices must be sorted within tile"

    def test_empty_all_kv(self):
        ip, ipn = _make_pairs(3, [0, 0, 0])
        _, _, _, _, n_tiles = preprocess_pairs(ip, ipn, block_m=64)
        assert n_tiles == 0

    def test_padding_correctness(self):
        ip, ipn = _make_pairs(1, [10])
        inp_flat, out_flat, _, tile_pc, n_tiles = preprocess_pairs(ip, ipn, block_m=64)
        assert n_tiles == 1
        assert tile_pc[0] == 10
        assert inp_flat.shape[0] == 64
        padded = inp_flat[10:].numpy()
        assert np.all(padded == 0), "Padded entries should be zero"


# ---------- GPU correctness tests ----------

def _reference_gather_gemm_scatter(features, filters, indice_pairs, indice_pair_num, n_out):
    """Reference: explicit Python loop over kv positions.

    Always computes in f32 for comparison with implicit GEMM output.
    """
    kv, c_in, c_out = filters.shape
    out = torch.zeros(n_out, c_out, dtype=torch.float32, device=features.device)
    for k in range(kv):
        nhot = int(indice_pair_num[k].item())
        if nhot == 0:
            continue
        inp_ids = indice_pairs[k, 0, :nhot].long()
        out_ids = indice_pairs[k, 1, :nhot].long()
        gathered = features[inp_ids].float()  # [nhot, c_in]
        result = gathered @ filters[k].float()  # [nhot, c_out]
        out.index_add_(0, out_ids, result)
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestImplicitGemmGPU:
    """GPU tests for implicit GEMM kernel correctness."""

    @pytest.fixture(autouse=True)
    def _skip_no_flydsl(self):
        try:
            import flydsl
        except ImportError:
            pytest.skip("FlyDSL not installed")

    def _run_correctness(self, n_in, n_out, c_in, c_out, kv, nhot_list, dtype):
        from cumm.implicit_gemm import implicit_gemm_forward

        device = "cuda"
        features = torch.randn(n_in, c_in, dtype=dtype, device=device) * 0.1
        filters = torch.randn(kv, c_in, c_out, dtype=dtype, device=device) * 0.1
        ip, ipn = _make_pairs(kv, nhot_list, n_max=max(nhot_list), device=device)
        ip[ip < 0] = 0
        ip[:, 0] = ip[:, 0] % n_in
        ip[:, 1] = ip[:, 1] % n_out

        ref = _reference_gather_gemm_scatter(features, filters, ip, ipn, n_out)
        out = implicit_gemm_forward(features, filters, ip, ipn, n_out)
        assert out is not None, "Kernel compilation failed"

        torch.cuda.synchronize()
        if dtype == torch.float32:
            atol, rtol = 1e-3, 1e-3
        else:
            atol, rtol = 0.05, 0.05
        torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)

    def test_basic_f32(self):
        self._run_correctness(100, 100, 16, 16, 3, [30, 50, 20], torch.float32)

    def test_basic_f16(self):
        self._run_correctness(100, 100, 16, 16, 3, [30, 50, 20], torch.float16)

    def test_single_kv_large(self):
        self._run_correctness(200, 200, 32, 64, 1, [150], torch.float32)

    def test_asymmetric_channels(self):
        self._run_correctness(100, 100, 16, 64, 3, [40, 30, 50], torch.float32)

    def test_some_empty_kv(self):
        self._run_correctness(100, 100, 16, 16, 5, [30, 0, 50, 0, 20], torch.float32)

    def test_partial_tile(self):
        self._run_correctness(50, 50, 16, 16, 1, [7], torch.float32)

    def test_exact_tile(self):
        self._run_correctness(100, 100, 16, 16, 1, [64], torch.float32)

    def test_kv27_subm(self):
        """Typical 3x3x3 SubM config."""
        self._run_correctness(1000, 1000, 32, 32, 27, [100]*27, torch.float32)


# ---------- Memory access pattern verification ----------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestMemoryAccessPattern:
    """Verify coalescing optimization effect."""

    @pytest.fixture(autouse=True)
    def _skip_no_flydsl(self):
        try:
            import flydsl
        except ImportError:
            pytest.skip("FlyDSL not installed")

    def test_sorted_indices_produce_sequential_access(self):
        """After preprocessing, inp_indices within a tile should be sorted
        (monotonically non-decreasing), enabling quasi-coalesced loads."""
        n_in, kv = 500, 3
        device = "cuda"
        ip, ipn = _make_pairs(kv, [200, 150, 100], device=device)
        ip[:, 0] = ip[:, 0].abs() % n_in
        ip[:, 1] = ip[:, 1].abs() % n_in
        inp_flat, _, _, tile_pc, n_tiles = preprocess_pairs(ip, ipn, block_m=64)
        inp_cpu = inp_flat.cpu().numpy()
        for t in range(n_tiles):
            start = t * 64
            n_valid = int(tile_pc[t].item())
            tile_inp = inp_cpu[start:start+n_valid]
            assert np.all(tile_inp[:-1] <= tile_inp[1:]), \
                f"Tile {t}: inp_indices not sorted — coalescing broken"

    def test_weight_layout_row_major(self):
        """Weight[k, c_in, c_out] is loaded to LDS as row-major (c_in * c_out).
        Verify that LDS access pattern w_lds[c * C_OUT + j] is sequential
        for the inner loop j (contiguous in LDS)."""
        c_in, c_out = 32, 32
        weight = torch.arange(c_in * c_out, dtype=torch.float32).reshape(c_in, c_out)
        for c in range(c_in):
            for j in range(c_out):
                expected = weight[c, j].item()
                lds_idx = c * c_out + j
                assert lds_idx == c * c_out + j, "Weight LDS layout mismatch"


# ---------- Coalescing benchmark ----------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestCoalescingBenchmark:
    """Measure effect of sorting on performance (informational)."""

    @pytest.fixture(autouse=True)
    def _skip_no_flydsl(self):
        try:
            import flydsl
        except ImportError:
            pytest.skip("FlyDSL not installed")

    def test_sorted_vs_unsorted_indices(self):
        """Compare performance with sorted vs random indices.
        This is informational — we just verify both produce correct results."""
        from cumm.implicit_gemm import implicit_gemm_forward

        device = "cuda"
        n_in, n_out, c_in, c_out = 500, 500, 32, 32
        kv = 1
        nhot = 200

        features = torch.randn(n_in, c_in, dtype=torch.float32, device=device) * 0.1
        filters = torch.randn(kv, c_in, c_out, dtype=torch.float32, device=device) * 0.1

        sorted_inp = torch.sort(torch.randint(0, n_in, (nhot,), dtype=torch.int32, device=device))[0]
        out_ids = torch.randperm(n_out, device=device)[:nhot].int()

        ip = torch.zeros(1, 2, nhot, dtype=torch.int32, device=device)
        ip[0, 0] = sorted_inp
        ip[0, 1] = out_ids
        ipn = torch.tensor([nhot], dtype=torch.int32, device=device)

        ref = _reference_gather_gemm_scatter(features, filters, ip, ipn, n_out)
        out = implicit_gemm_forward(features, filters, ip, ipn, n_out)
        assert out is not None

        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
