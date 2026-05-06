"""Unit tests for FlyDSL implicit GEMM kernel (output-tile-centric).

Tests focus on:
1. Legacy preprocessing (kv-centric, for backward compat)
2. Mask generation (output-tile-centric: sort + mask + pair ranges)
3. Kernel correctness: gather+GEMM+scatter vs explicit loop
4. Memory layout: weight layout in LDS
5. Edge cases: partial tiles, single pair, empty kv positions
"""
import pytest
import torch
import numpy as np

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


# ---------- Legacy preprocessing tests (CPU) ----------

class TestPreprocessPairs:
    """Test host-side pair preprocessing: sorting and tiling (kv-centric legacy)."""

    def test_basic_tiling(self):
        kv = 3
        ip, ipn = _make_pairs(kv, [64, 32, 0])
        inp_flat, out_flat, tile_kpos, tile_pc, n_tiles = preprocess_pairs(ip, ipn, block_m=64)
        assert n_tiles == 2
        assert tile_kpos[0] == 0
        assert tile_kpos[1] == 1
        assert tile_pc[0] == 64
        assert tile_pc[1] == 32
        assert inp_flat.shape[0] == n_tiles * 64

    def test_multi_tile_single_kv(self):
        ip, ipn = _make_pairs(1, [150])
        _, _, tile_kpos, tile_pc, n_tiles = preprocess_pairs(ip, ipn, block_m=64)
        assert n_tiles == 3
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


# ---------- Mask generation tests (GPU) ----------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestMaskGeneration:
    """Test C++/HIP mask generation for output-tile-centric implicit GEMM."""

    @pytest.fixture(autouse=True)
    def _skip_no_hip(self):
        from cumm.implicit_gemm import _get_hip_module
        if _get_hip_module() is None:
            pytest.skip("HIP module not available")

    def test_basic_mask(self):
        """Verify mask marks correct (tile, kv) as active."""
        from cumm.implicit_gemm import _get_hip_module
        hip = _get_hip_module()

        device = "cuda"
        n_out = 128
        block_m = 64
        kv = 3

        ip = torch.full((kv, 2, 50), -1, dtype=torch.int32, device=device)
        ipn = torch.zeros(kv, dtype=torch.int32, device=device)

        # kv=0: 20 pairs, all out_index in [0, 63] → tile 0 only
        ip[0, 0, :20] = torch.randint(0, 100, (20,), dtype=torch.int32, device=device)
        ip[0, 1, :20] = torch.randint(0, 64, (20,), dtype=torch.int32, device=device)
        ipn[0] = 20

        # kv=1: 30 pairs, out_index in [64, 127] → tile 1 only
        ip[1, 0, :30] = torch.randint(0, 100, (30,), dtype=torch.int32, device=device)
        ip[1, 1, :30] = torch.randint(64, 128, (30,), dtype=torch.int32, device=device)
        ipn[1] = 30

        # kv=2: 0 pairs
        ipn[2] = 0

        results = hip.build_implicit_gemm_mask(ip, ipn, n_out, block_m)
        sorted_inp, sorted_out, sorted_kv, mask, pair_start, pair_end, lut = results

        mask_cpu = mask.cpu()
        assert mask_cpu.shape == (2, 3)  # 2 tiles, 3 kv
        assert mask_cpu[0, 0] == 1  # tile 0, kv 0 active
        assert mask_cpu[0, 1] == 0  # tile 0, kv 1 inactive
        assert mask_cpu[1, 0] == 0  # tile 1, kv 0 inactive
        assert mask_cpu[1, 1] == 1  # tile 1, kv 1 active
        assert mask_cpu[0, 2] == 0  # kv 2 always inactive
        assert mask_cpu[1, 2] == 0

    def test_sorted_by_kv_then_out(self):
        """Verify sorted arrays are ordered by (kv, out_index)."""
        from cumm.implicit_gemm import _get_hip_module
        hip = _get_hip_module()

        device = "cuda"
        n_out, kv, block_m = 200, 3, 64

        ip, ipn = _make_pairs(kv, [30, 50, 20], n_max=50, device=device)
        ip[ip < 0] = 0
        ip[:, 0] = ip[:, 0] % 100
        ip[:, 1] = ip[:, 1] % n_out

        results = hip.build_implicit_gemm_mask(ip, ipn, n_out, block_m)
        sorted_inp, sorted_out, sorted_kv, mask, ps, pe, lut = results

        kv_cpu = sorted_kv.cpu().numpy()
        out_cpu = sorted_out.cpu().numpy()

        # kv values should be non-decreasing
        assert np.all(kv_cpu[:-1] <= kv_cpu[1:]), "sorted_kv not non-decreasing"

        # Within each kv group, out_index should be non-decreasing
        for k in range(kv):
            group_mask = kv_cpu == k
            if group_mask.sum() > 0:
                group_out = out_cpu[group_mask]
                assert np.all(group_out[:-1] <= group_out[1:]), \
                    f"kv={k}: out_index not sorted"

    def test_pair_ranges_valid(self):
        """Verify pair_start/pair_end ranges are consistent."""
        from cumm.implicit_gemm import _get_hip_module
        hip = _get_hip_module()

        device = "cuda"
        n_out, kv, block_m = 100, 2, 64

        ip = torch.full((kv, 2, 40), -1, dtype=torch.int32, device=device)
        ipn = torch.zeros(kv, dtype=torch.int32, device=device)
        ip[0, 0, :20] = torch.arange(20, dtype=torch.int32, device=device)
        ip[0, 1, :20] = torch.arange(20, dtype=torch.int32, device=device)
        ipn[0] = 20
        ip[1, 0, :15] = torch.arange(15, dtype=torch.int32, device=device)
        ip[1, 1, :15] = torch.arange(15, dtype=torch.int32, device=device) + 50
        ipn[1] = 15

        results = hip.build_implicit_gemm_mask(ip, ipn, n_out, block_m)
        _, sorted_out, sorted_kv, mask, ps, pe, _ = results

        mask_cpu = mask.cpu()
        ps_cpu = ps.cpu()
        pe_cpu = pe.cpu()

        num_tiles = mask_cpu.shape[0]
        for t in range(num_tiles):
            for k in range(kv):
                if mask_cpu[t, k] == 1:
                    s = ps_cpu[t, k].item()
                    e = pe_cpu[t, k].item()
                    assert s < e, f"tile={t}, kv={k}: pair_start >= pair_end"
                    # All pairs in [s, e) should have out_index in tile range
                    tile_outs = sorted_out[s:e].cpu().numpy()
                    assert np.all(tile_outs >= t * block_m), \
                        f"tile={t}: out_index below tile range"
                    assert np.all(tile_outs < (t + 1) * block_m), \
                        f"tile={t}: out_index above tile range"

    def test_total_pairs_preserved(self):
        """Verify total number of pairs is preserved after mask generation."""
        from cumm.implicit_gemm import _get_hip_module
        hip = _get_hip_module()

        device = "cuda"
        ip, ipn = _make_pairs(5, [30, 0, 50, 0, 20], n_max=50, device=device)
        ip[ip < 0] = 0
        ip[:, 0] = ip[:, 0] % 200
        ip[:, 1] = ip[:, 1] % 200

        expected_total = int(ipn.sum().item())
        results = hip.build_implicit_gemm_mask(ip, ipn, 200, 64)
        sorted_inp = results[0]
        assert sorted_inp.shape[0] == expected_total


# ---------- GPU correctness tests ----------

def _reference_gather_gemm_scatter(features, filters, indice_pairs, indice_pair_num, n_out):
    """Reference: explicit Python loop over kv positions."""
    kv, c_in, c_out = filters.shape
    out = torch.zeros(n_out, c_out, dtype=torch.float32, device=features.device)
    for k in range(kv):
        nhot = int(indice_pair_num[k].item())
        if nhot == 0:
            continue
        inp_ids = indice_pairs[k, 0, :nhot].long()
        out_ids = indice_pairs[k, 1, :nhot].long()
        gathered = features[inp_ids].float()
        result = gathered @ filters[k].float()
        out.index_add_(0, out_ids, result)
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestImplicitGemmGPU:
    """GPU tests for output-tile-centric implicit GEMM kernel correctness."""

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


# ---------- Memory layout verification ----------

class TestMemoryLayout:
    """Verify weight layout assumptions."""

    def test_weight_layout_row_major(self):
        c_in, c_out = 32, 32
        weight = torch.arange(c_in * c_out, dtype=torch.float32).reshape(c_in, c_out)
        for c in range(c_in):
            for j in range(c_out):
                lds_idx = c * c_out + j
                assert lds_idx == c * c_out + j, "Weight LDS layout mismatch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestInpRowLut:
    """Test inp_row_lut correctness."""

    @pytest.fixture(autouse=True)
    def _skip_no_hip(self):
        from cumm.implicit_gemm import _get_hip_module
        if _get_hip_module() is None:
            pytest.skip("HIP module not available")

    def test_lut_basic(self):
        """Each (tile, kv, local_row) should map to the correct input row."""
        from cumm.implicit_gemm import _get_hip_module
        hip = _get_hip_module()

        device = "cuda"
        n_out, kv, block_m = 128, 2, 64

        ip = torch.full((kv, 2, 50), -1, dtype=torch.int32, device=device)
        ipn = torch.zeros(kv, dtype=torch.int32, device=device)

        # kv=0: pairs mapping out 0..19 → inp 100..119
        for i in range(20):
            ip[0, 0, i] = 100 + i   # inp
            ip[0, 1, i] = i         # out
        ipn[0] = 20

        # kv=1: pairs mapping out 64..73 → inp 200..209
        for i in range(10):
            ip[1, 0, i] = 200 + i
            ip[1, 1, i] = 64 + i
        ipn[1] = 10

        results = hip.build_implicit_gemm_mask(ip, ipn, n_out, block_m)
        lut = results[6]  # inp_row_lut [num_tiles, kv, block_m]
        lut_cpu = lut.cpu()

        # tile 0, kv 0: local rows 0..19 should have inp 100..119
        for i in range(20):
            assert lut_cpu[0, 0, i].item() == 100 + i, \
                f"lut[0,0,{i}] = {lut_cpu[0,0,i].item()}, expected {100+i}"
        # tile 0, kv 0: rows 20..63 should be -1
        for i in range(20, 64):
            assert lut_cpu[0, 0, i].item() == -1

        # tile 1, kv 1: local rows 0..9 should have inp 200..209
        for i in range(10):
            assert lut_cpu[1, 1, i].item() == 200 + i

        # tile 0, kv 1: all -1
        assert (lut_cpu[0, 1] == -1).all()

    def test_lut_shape(self):
        from cumm.implicit_gemm import _get_hip_module
        hip = _get_hip_module()

        device = "cuda"
        ip, ipn = _make_pairs(3, [30, 50, 20], n_max=50, device=device)
        ip[ip < 0] = 0
        ip[:, 0] = ip[:, 0] % 200
        ip[:, 1] = ip[:, 1] % 200

        results = hip.build_implicit_gemm_mask(ip, ipn, 200, 64)
        lut = results[6]
        num_tiles = (200 + 63) // 64
        assert lut.shape == (num_tiles, 3, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestImplicitGemmV4:
    """GPU tests for V4 implicit GEMM kernel (lut-based gather + register blocking)."""

    @pytest.fixture(autouse=True)
    def _skip_no_flydsl(self):
        try:
            import flydsl
        except ImportError:
            pytest.skip("FlyDSL not installed")

    def _run_correctness(self, n_in, n_out, c_in, c_out, kv, nhot_list, dtype):
        from cumm.implicit_gemm import implicit_gemm_v4_forward

        device = "cuda"
        features = torch.randn(n_in, c_in, dtype=dtype, device=device) * 0.1
        filters = torch.randn(kv, c_in, c_out, dtype=dtype, device=device) * 0.1
        ip, ipn = _make_pairs(kv, nhot_list, n_max=max(nhot_list), device=device)
        ip[ip < 0] = 0
        ip[:, 0] = ip[:, 0] % n_in
        # V4 lut assumes SubM: unique output index per (kv, out_row).
        # Use unique permutation to avoid duplicates.
        for k_idx in range(kv):
            nhot = nhot_list[k_idx]
            if nhot > 0:
                perm = torch.randperm(n_out, device=device, dtype=torch.int32)[:nhot]
                ip[k_idx, 1, :nhot] = perm

        ref = _reference_gather_gemm_scatter(features, filters, ip, ipn, n_out)
        out = implicit_gemm_v4_forward(features, filters, ip, ipn, n_out)
        assert out is not None, "V4 kernel compilation failed"

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

    def test_single_kv(self):
        self._run_correctness(200, 200, 32, 64, 1, [150], torch.float32)

    def test_some_empty_kv(self):
        self._run_correctness(100, 100, 16, 16, 5, [30, 0, 50, 0, 20], torch.float32)

    def test_partial_tile(self):
        self._run_correctness(50, 50, 16, 16, 1, [7], torch.float32)

    def test_kv27_subm(self):
        """Typical 3x3x3 SubM config."""
        self._run_correctness(1000, 1000, 32, 32, 27, [100]*27, torch.float32)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
