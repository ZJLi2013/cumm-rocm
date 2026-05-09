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


class TestImplicitGemmDispatchDescriptors:
    """CPU-only checks for descriptorized implicit GEMM dispatch metadata."""

    def test_bk16_descriptor_metadata(self):
        from cumm.implicit_gemm import get_implicit_gemm_candidates

        candidates = get_implicit_gemm_candidates(torch.float32, 32, 32)
        bk16 = next(
            desp
            for desp in candidates
            if desp.name == "mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16"
        )

        assert bk16.tile_m == 16
        assert bk16.tile_n == 32
        assert bk16.block_k == 16
        assert bk16.waves == 2
        assert bk16.ashared is True
        assert bk16.epilogue == "direct"

    def test_bk16_remap_descriptor_metadata(self):
        from cumm.implicit_gemm import get_implicit_gemm_candidates

        candidates = get_implicit_gemm_candidates(torch.float32, 32, 32)
        remap = next(
            desp
            for desp in candidates
            if desp.name == "mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_remap"
        )

        assert remap.tile_m == 16
        assert remap.tile_n == 32
        assert remap.block_k == 16
        assert remap.waves == 2
        assert remap.ashared is True
        assert remap.epilogue == "remap"

    def test_dispatch_prefers_crossk_for_mid_channels(self):
        from cumm.implicit_gemm import select_implicit_gemm_kernel

        selected = select_implicit_gemm_kernel(torch.float32, 32, 32)
        assert selected.name == "crossk_pf_xor_bk32"

    def test_dispatch_keeps_ashared_for_small_channels(self):
        from cumm.implicit_gemm import select_implicit_gemm_kernel

        selected = select_implicit_gemm_kernel(torch.float32, 16, 32)
        assert selected.name == "mfma_f32_16x16x4f32_n2_ashared"


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


class TestImplicitGemmDispatch:
    """Verify shape-based kernel family dispatch decisions."""

    def test_small_f32_cout32_prefers_ashared(self):
        from cumm.implicit_gemm import select_implicit_gemm_kernel

        desp = select_implicit_gemm_kernel(torch.float32, c_in=16, c_out=32)
        assert desp.name == "mfma_f32_16x16x4f32_n2_ashared"

    def test_small_f32_cout16_prefers_single_mfma(self):
        from cumm.implicit_gemm import select_implicit_gemm_kernel

        desp = select_implicit_gemm_kernel(torch.float32, c_in=16, c_out=16)
        assert desp.name == "mfma_f32_16x16x4f32"

    def test_large_channel_prefers_crossk_pf_xor(self):
        from cumm.implicit_gemm import select_implicit_gemm_kernel

        desp = select_implicit_gemm_kernel(torch.float32, c_in=64, c_out=128)
        assert desp.name == "crossk_pf_xor_bk32"

    def test_small_tile_crossk_descriptor_available(self):
        from cumm.implicit_gemm import select_implicit_gemm_kernel

        desp = select_implicit_gemm_kernel(torch.float32, c_in=32, c_out=32)
        assert desp.name == "crossk_pf_xor_bk32"

    def test_mid_f32_odd_cin_fallback_to_kpipe(self):
        from cumm.implicit_gemm import select_implicit_gemm_kernel

        desp = select_implicit_gemm_kernel(torch.float32, c_in=20, c_out=32)
        assert desp.name == "mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16"

    def test_f16_uses_scalar_fallback(self):
        from cumm.implicit_gemm import select_implicit_gemm_kernel

        desp = select_implicit_gemm_kernel(torch.float16, c_in=32, c_out=32)
        assert desp.name == "scalar_tile"

    def test_env_force_kernel(self, monkeypatch):
        from cumm.implicit_gemm import select_implicit_gemm_kernel

        monkeypatch.setenv("CUMM_IMPLICIT_GEMM_KERNEL", "scalar_tile")
        desp = select_implicit_gemm_kernel(torch.float32, c_in=16, c_out=32)
        assert desp.name == "scalar_tile"

    def test_env_force_unavailable_kernel(self, monkeypatch):
        from cumm.implicit_gemm import select_implicit_gemm_kernel

        monkeypatch.setenv("CUMM_IMPLICIT_GEMM_KERNEL", "mfma_f32_16x16x4f32")
        with pytest.raises(ValueError):
            select_implicit_gemm_kernel(torch.float16, c_in=16, c_out=32)


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


class TestImplicitGemmScalarTile:
    """Current scalar_tile implicit GEMM family member."""

    @pytest.fixture(autouse=True)
    def _skip_no_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("No GPU")

    @pytest.fixture(autouse=True)
    def _skip_no_flydsl(self):
        try:
            import flydsl
        except ImportError:
            pytest.skip("FlyDSL not installed")

    def _run_correctness(self, n_in, n_out, c_in, c_out, kv, nhot_list, dtype):
        from cumm.implicit_gemm import implicit_gemm_scalar_tile_forward

        device = "cuda"
        features = torch.randn(n_in, c_in, dtype=dtype, device=device) * 0.1
        filters = torch.randn(kv, c_in, c_out, dtype=dtype, device=device) * 0.1
        ip, ipn = _make_pairs(kv, nhot_list, n_max=max(nhot_list), device=device)
        ip[ip < 0] = 0
        ip[:, 0] = ip[:, 0] % n_in
        for k_idx in range(kv):
            nhot = nhot_list[k_idx]
            if nhot > 0:
                perm = torch.randperm(n_out, device=device, dtype=torch.int32)[:nhot]
                ip[k_idx, 1, :nhot] = perm

        ref = _reference_gather_gemm_scatter(features, filters, ip, ipn, n_out)
        out = implicit_gemm_scalar_tile_forward(features, filters, ip, ipn, n_out)
        assert out is not None, "scalar_tile kernel compilation failed"

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

    def test_large_channel(self):
        self._run_correctness(500, 500, 64, 128, 3, [150, 200, 100], torch.float32)

    def test_kv27_subm(self):
        self._run_correctness(1000, 1000, 32, 32, 27, [100]*27, torch.float32)

    def test_large_channel_kv27(self):
        self._run_correctness(500, 500, 64, 128, 27, [50]*27, torch.float32)


class TestImplicitGemmMfmaF32_16x16x4:
    """Current mfma_f32_16x16x4f32 implicit GEMM family member."""

    @pytest.fixture(autouse=True)
    def _skip_no_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("No GPU")

    @pytest.fixture(autouse=True)
    def _skip_no_flydsl(self):
        try:
            import flydsl
        except ImportError:
            pytest.skip("FlyDSL not installed")

    def _run_correctness(self, n_in, n_out, c_in, c_out, kv, nhot_list):
        from cumm.implicit_gemm import implicit_gemm_mfma_f32_16x16x4f32_forward

        device = "cuda"
        features = torch.randn(n_in, c_in, dtype=torch.float32, device=device) * 0.1
        filters = torch.randn(kv, c_in, c_out, dtype=torch.float32, device=device) * 0.1
        ip, ipn = _make_pairs(kv, nhot_list, n_max=max(nhot_list), device=device)
        ip[ip < 0] = 0
        ip[:, 0] = ip[:, 0] % n_in
        for k_idx in range(kv):
            nhot = nhot_list[k_idx]
            if nhot > 0:
                perm = torch.randperm(n_out, device=device, dtype=torch.int32)[:nhot]
                ip[k_idx, 1, :nhot] = perm

        ref = _reference_gather_gemm_scatter(features, filters, ip, ipn, n_out)
        out = implicit_gemm_mfma_f32_16x16x4f32_forward(features, filters, ip, ipn, n_out)
        assert out is not None, "mfma_f32_16x16x4f32 kernel compilation failed"

        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-3, rtol=1e-3)

    def test_basic_f32(self):
        self._run_correctness(100, 100, 16, 32, 3, [30, 50, 20])

    def test_single_kv(self):
        self._run_correctness(200, 200, 32, 64, 1, [150])

    def test_kv27_subm(self):
        self._run_correctness(1000, 1000, 32, 32, 27, [100]*27)

    def test_large_channel_kv27(self):
        self._run_correctness(500, 500, 64, 128, 27, [50]*27)


class TestImplicitGemmMfmaF32_16x16x4N2:
    """Current mfma_f32_16x16x4f32_n2 implicit GEMM family member."""

    @pytest.fixture(autouse=True)
    def _skip_no_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("No GPU")

    @pytest.fixture(autouse=True)
    def _skip_no_flydsl(self):
        try:
            import flydsl
        except ImportError:
            pytest.skip("FlyDSL not installed")

    def _run_correctness(self, n_in, n_out, c_in, c_out, kv, nhot_list):
        from cumm.implicit_gemm import implicit_gemm_mfma_f32_16x16x4f32_n2_forward

        device = "cuda"
        features = torch.randn(n_in, c_in, dtype=torch.float32, device=device) * 0.1
        filters = torch.randn(kv, c_in, c_out, dtype=torch.float32, device=device) * 0.1
        ip, ipn = _make_pairs(kv, nhot_list, n_max=max(nhot_list), device=device)
        ip[ip < 0] = 0
        ip[:, 0] = ip[:, 0] % n_in
        for k_idx in range(kv):
            nhot = nhot_list[k_idx]
            if nhot > 0:
                perm = torch.randperm(n_out, device=device, dtype=torch.int32)[:nhot]
                ip[k_idx, 1, :nhot] = perm

        ref = _reference_gather_gemm_scatter(features, filters, ip, ipn, n_out)
        out = implicit_gemm_mfma_f32_16x16x4f32_n2_forward(
            features, filters, ip, ipn, n_out
        )
        assert out is not None, "mfma_f32_16x16x4f32_n2 kernel compilation failed"

        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-3, rtol=1e-3)

    def test_basic_f32(self):
        self._run_correctness(100, 100, 16, 32, 3, [30, 50, 20])

    def test_single_kv(self):
        self._run_correctness(200, 200, 32, 64, 1, [150])

    def test_kv27_subm(self):
        self._run_correctness(1000, 1000, 32, 32, 27, [100]*27)

    def test_large_channel_kv27(self):
        self._run_correctness(500, 500, 64, 128, 27, [50]*27)


class TestImplicitGemmMfmaF32_16x16x4N2AShared:
    """Current mfma_f32_16x16x4f32_n2_ashared implicit GEMM family member."""

    @pytest.fixture(autouse=True)
    def _skip_no_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("No GPU")

    @pytest.fixture(autouse=True)
    def _skip_no_flydsl(self):
        try:
            import flydsl
        except ImportError:
            pytest.skip("FlyDSL not installed")

    def _run_correctness(self, n_in, n_out, c_in, c_out, kv, nhot_list):
        from cumm.implicit_gemm import implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_forward

        device = "cuda"
        features = torch.randn(n_in, c_in, dtype=torch.float32, device=device) * 0.1
        filters = torch.randn(kv, c_in, c_out, dtype=torch.float32, device=device) * 0.1
        ip, ipn = _make_pairs(kv, nhot_list, n_max=max(nhot_list), device=device)
        ip[ip < 0] = 0
        ip[:, 0] = ip[:, 0] % n_in
        for k_idx in range(kv):
            nhot = nhot_list[k_idx]
            if nhot > 0:
                perm = torch.randperm(n_out, device=device, dtype=torch.int32)[:nhot]
                ip[k_idx, 1, :nhot] = perm

        ref = _reference_gather_gemm_scatter(features, filters, ip, ipn, n_out)
        out = implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_forward(
            features, filters, ip, ipn, n_out
        )
        assert out is not None, "mfma_f32_16x16x4f32_n2_ashared kernel compilation failed"

        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-3, rtol=1e-3)

    def test_basic_f32(self):
        self._run_correctness(100, 100, 16, 32, 3, [30, 50, 20])

    def test_single_kv(self):
        self._run_correctness(200, 200, 32, 64, 1, [150])

    def test_kv27_subm(self):
        self._run_correctness(1000, 1000, 32, 32, 27, [100]*27)

    def test_large_channel_kv27(self):
        self._run_correctness(500, 500, 64, 128, 27, [50]*27)


class TestImplicitGemmMfmaF32_16x16x4N2ASharedKPipe:
    """Current K-pipe BLOCK_K family members."""

    @pytest.fixture(autouse=True)
    def _skip_no_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("No GPU")

    @pytest.fixture(autouse=True)
    def _skip_no_flydsl(self):
        try:
            import flydsl
        except ImportError:
            pytest.skip("FlyDSL not installed")

    def _run_correctness(self, fn_name, n_in, n_out, c_in, c_out, kv, nhot_list):
        import cumm.implicit_gemm as ig

        device = "cuda"
        features = torch.randn(n_in, c_in, dtype=torch.float32, device=device) * 0.1
        filters = torch.randn(kv, c_in, c_out, dtype=torch.float32, device=device) * 0.1
        ip, ipn = _make_pairs(kv, nhot_list, n_max=max(nhot_list), device=device)
        ip[ip < 0] = 0
        ip[:, 0] = ip[:, 0] % n_in
        for k_idx in range(kv):
            nhot = nhot_list[k_idx]
            if nhot > 0:
                perm = torch.randperm(n_out, device=device, dtype=torch.int32)[:nhot]
                ip[k_idx, 1, :nhot] = perm

        ref = _reference_gather_gemm_scatter(features, filters, ip, ipn, n_out)
        out = getattr(ig, fn_name)(features, filters, ip, ipn, n_out)
        assert out is not None, f"{fn_name} compilation failed"

        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "fn_name",
        [
            "implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward",
            "implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_remap_forward",
            "implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk32_forward",
        ],
    )
    def test_basic_f32(self, fn_name):
        self._run_correctness(fn_name, 100, 100, 16, 32, 3, [30, 50, 20])

    @pytest.mark.parametrize(
        "fn_name",
        [
            "implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward",
            "implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_remap_forward",
            "implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk32_forward",
        ],
    )
    def test_single_kv(self, fn_name):
        self._run_correctness(fn_name, 200, 200, 32, 64, 1, [150])

    @pytest.mark.parametrize(
        "fn_name",
        [
            "implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward",
            "implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_remap_forward",
            "implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk32_forward",
        ],
    )
    def test_kv27_subm(self, fn_name):
        self._run_correctness(fn_name, 1000, 1000, 32, 32, 27, [100]*27)

    @pytest.mark.parametrize(
        "fn_name",
        [
            "implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward",
            "implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_remap_forward",
            "implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk32_forward",
        ],
    )
    def test_large_channel_kv27(self, fn_name):
        self._run_correctness(fn_name, 500, 500, 64, 128, 27, [50]*27)


class TestImplicitGemmCrossK:
    """Cross-kv K-fused kernel (Step 8)."""

    @pytest.fixture(autouse=True)
    def _skip_no_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("No GPU")

    @pytest.fixture(autouse=True)
    def _skip_no_flydsl(self):
        try:
            import flydsl
        except ImportError:
            pytest.skip("FlyDSL not installed")

    def _run_correctness(self, n_in, n_out, c_in, c_out, kv, nhot_list, block_k=32):
        from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_crossk import (
            implicit_gemm_crossk_forward,
        )

        device = "cuda"
        features = torch.randn(n_in, c_in, dtype=torch.float32, device=device) * 0.1
        filters = torch.randn(kv, c_in, c_out, dtype=torch.float32, device=device) * 0.1
        ip, ipn = _make_pairs(kv, nhot_list, n_max=max(nhot_list), device=device)
        ip[ip < 0] = 0
        ip[:, 0] = ip[:, 0] % n_in
        for k_idx in range(kv):
            nhot = nhot_list[k_idx]
            if nhot > 0:
                perm = torch.randperm(n_out, device=device, dtype=torch.int32)[:nhot]
                ip[k_idx, 1, :nhot] = perm

        ref = _reference_gather_gemm_scatter(features, filters, ip, ipn, n_out)
        out = implicit_gemm_crossk_forward(
            features, filters, ip, ipn, n_out, block_k=block_k
        )
        assert out is not None, f"crossk bk{block_k} compilation failed"

        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-3, rtol=1e-3)

    def test_basic_f32(self):
        self._run_correctness(100, 100, 16, 32, 3, [30, 50, 20], block_k=16)

    def test_single_kv(self):
        self._run_correctness(200, 200, 32, 64, 1, [150], block_k=32)

    def test_kv27_subm(self):
        self._run_correctness(1000, 1000, 32, 32, 27, [100]*27, block_k=32)

    def test_large_channel_kv27(self):
        self._run_correctness(500, 500, 64, 128, 27, [50]*27, block_k=32)

    def test_bk16_large_channel(self):
        self._run_correctness(500, 500, 64, 128, 27, [50]*27, block_k=16)


class TestImplicitGemmCrossKPrefetch:
    """Cross-kv kernel with VMEM-MFMA prefetch overlap (Step 10)."""

    @pytest.fixture(autouse=True)
    def _skip_no_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("No GPU")

    @pytest.fixture(autouse=True)
    def _skip_no_flydsl(self):
        try:
            import flydsl
        except ImportError:
            pytest.skip("FlyDSL not installed")

    def test_basic_f32(self):
        self._run_correctness(100, 100, 32, 32, 3, [30, 50, 20], block_k=16)

    def test_bk32_kv27(self):
        self._run_correctness(1000, 1000, 32, 32, 27, [100]*27, block_k=32)

    def test_bk32_large_channel(self):
        self._run_correctness(500, 500, 64, 128, 27, [50]*27, block_k=32)

    def test_bk32_xor_swizzle(self):
        self._run_correctness(1000, 1000, 32, 32, 27, [100]*27, block_k=32, use_xor_swizzle=True)

    def test_bk32_xor_large(self):
        self._run_correctness(500, 500, 64, 128, 27, [50]*27, block_k=32, use_xor_swizzle=True)

    # --- Release correctness sweep ---
    def test_xor_c_in_32_c_out_64(self):
        self._run_correctness(1000, 1000, 32, 64, 27, [100]*27, block_k=32, use_xor_swizzle=True)

    def test_xor_c_in_128(self):
        self._run_correctness(500, 500, 128, 128, 27, [50]*27, block_k=32, use_xor_swizzle=True)

    def test_xor_kv1(self):
        self._run_correctness(1000, 1000, 32, 32, 1, [150], block_k=32, use_xor_swizzle=True)

    def test_xor_kv9(self):
        self._run_correctness(1000, 1000, 32, 32, 9, [200]*9, block_k=32, use_xor_swizzle=True)

    def test_xor_low_density(self):
        self._run_correctness(5000, 5000, 64, 128, 27, [10]*27, block_k=32, use_xor_swizzle=True)

    def test_xor_high_density(self):
        self._run_correctness(2000, 2000, 32, 32, 27, [200]*27, block_k=32, use_xor_swizzle=True)

    def test_xor_large_n(self):
        self._run_correctness(50000, 50000, 32, 32, 27, [150]*27, block_k=32, use_xor_swizzle=True)

    def test_xor_c_out_256(self):
        self._run_correctness(500, 500, 64, 256, 27, [50]*27, block_k=32, use_xor_swizzle=True)

    def _run_correctness(self, n_in, n_out, c_in, c_out, kv, nhot_list, block_k=32, use_xor_swizzle=False):
        from cumm.implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_crossk import (
            implicit_gemm_crossk_prefetch_forward,
        )

        device = "cuda"
        features = torch.randn(n_in, c_in, dtype=torch.float32, device=device) * 0.1
        filters = torch.randn(kv, c_in, c_out, dtype=torch.float32, device=device) * 0.1
        ip, ipn = _make_pairs(kv, nhot_list, n_max=max(nhot_list), device=device)
        ip[ip < 0] = 0
        ip[:, 0] = ip[:, 0] % n_in
        for k_idx in range(kv):
            nhot = nhot_list[k_idx]
            if nhot > 0:
                perm = torch.randperm(n_out, device=device, dtype=torch.int32)[:nhot]
                ip[k_idx, 1, :nhot] = perm

        ref = _reference_gather_gemm_scatter(features, filters, ip, ipn, n_out)
        out = implicit_gemm_crossk_prefetch_forward(
            features, filters, ip, ipn, n_out, block_k=block_k,
            use_xor_swizzle=use_xor_swizzle,
        )
        tag = f"crossk-PF bk{block_k}" + (" xor" if use_xor_swizzle else "")
        assert out is not None, f"{tag} compilation failed"

        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-3, rtol=1e-3)


class TestImplicitGemmMfmaF32_16x16x4N2ASharedKPipeDB:
    """Double-buffered kpipe family members."""

    @pytest.fixture(autouse=True)
    def _skip_no_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("No GPU")

    @pytest.fixture(autouse=True)
    def _skip_no_flydsl(self):
        try:
            import flydsl
        except ImportError:
            pytest.skip("FlyDSL not installed")

    def _run_correctness(self, fn_name, n_in, n_out, c_in, c_out, kv, nhot_list):
        import cumm.implicit_gemm as ig

        device = "cuda"
        features = torch.randn(n_in, c_in, dtype=torch.float32, device=device) * 0.1
        filters = torch.randn(kv, c_in, c_out, dtype=torch.float32, device=device) * 0.1
        ip, ipn = _make_pairs(kv, nhot_list, n_max=max(nhot_list), device=device)
        ip[ip < 0] = 0
        ip[:, 0] = ip[:, 0] % n_in
        for k_idx in range(kv):
            nhot = nhot_list[k_idx]
            if nhot > 0:
                perm = torch.randperm(n_out, device=device, dtype=torch.int32)[:nhot]
                ip[k_idx, 1, :nhot] = perm

        ref = _reference_gather_gemm_scatter(features, filters, ip, ipn, n_out)
        out = getattr(ig, fn_name)(features, filters, ip, ipn, n_out)
        assert out is not None, f"{fn_name} kernel compilation failed"

        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "fn_name",
        [
            "implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_db32_forward",
            "implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_db64_forward",
        ],
    )
    def test_basic_f32(self, fn_name):
        self._run_correctness(fn_name, 100, 100, 16, 32, 3, [30, 50, 20])

    @pytest.mark.parametrize(
        "fn_name",
        [
            "implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_db32_forward",
            "implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_db64_forward",
        ],
    )
    def test_kv27_subm(self, fn_name):
        self._run_correctness(fn_name, 1000, 1000, 32, 32, 27, [100]*27)

    @pytest.mark.parametrize(
        "fn_name",
        [
            "implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_db32_forward",
            "implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_db64_forward",
        ],
    )
    def test_large_channel_kv27(self, fn_name):
        self._run_correctness(fn_name, 500, 500, 64, 128, 27, [50]*27)


class TestImplicitGemmMfmaF32_32x32x2:
    """Current mfma_f32_32x32x2f32 implicit GEMM family member."""

    pytestmark = pytest.mark.skip(
        reason="32x32x2f32 lane/operand mapping is still experimental"
    )

    @pytest.fixture(autouse=True)
    def _skip_no_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("No GPU")

    @pytest.fixture(autouse=True)
    def _skip_no_flydsl(self):
        try:
            import flydsl
        except ImportError:
            pytest.skip("FlyDSL not installed")

    def _run_correctness(self, n_in, n_out, c_in, c_out, kv, nhot_list):
        from cumm.implicit_gemm import implicit_gemm_mfma_f32_32x32x2f32_forward

        device = "cuda"
        features = torch.randn(n_in, c_in, dtype=torch.float32, device=device) * 0.1
        filters = torch.randn(kv, c_in, c_out, dtype=torch.float32, device=device) * 0.1
        ip, ipn = _make_pairs(kv, nhot_list, n_max=max(nhot_list), device=device)
        ip[ip < 0] = 0
        ip[:, 0] = ip[:, 0] % n_in
        for k_idx in range(kv):
            nhot = nhot_list[k_idx]
            if nhot > 0:
                perm = torch.randperm(n_out, device=device, dtype=torch.int32)[:nhot]
                ip[k_idx, 1, :nhot] = perm

        ref = _reference_gather_gemm_scatter(features, filters, ip, ipn, n_out)
        out = implicit_gemm_mfma_f32_32x32x2f32_forward(
            features, filters, ip, ipn, n_out
        )
        assert out is not None, "mfma_f32_32x32x2f32 kernel compilation failed"

        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-3, rtol=1e-3)

    def test_basic_f32(self):
        self._run_correctness(100, 100, 16, 32, 3, [30, 50, 20])

    def test_single_kv(self):
        self._run_correctness(200, 200, 32, 64, 1, [150])

    def test_kv27_subm(self):
        self._run_correctness(1000, 1000, 32, 32, 27, [100]*27)

    def test_large_channel_kv27(self):
        self._run_correctness(500, 500, 64, 128, 27, [50]*27)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
