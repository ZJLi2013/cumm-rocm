"""Test FlyDSLGemmTuner on ROCm GPU.

Run: python -m pytest test/test_gemm_tuner.py -v
Requires: FlyDSL installed, AMD GPU available.
"""

import pytest
import torch

from cumm.gemm_tuner import FlyDSLGemmTuner, _select_tile_config

# spconv typical shapes: (M, N, K)
SPCONV_SHAPES = [
    (200, 256, 256),
    (500, 128, 128),
    (1000, 256, 256),
    (2000, 128, 128),
    (5000, 256, 128),
    (4096, 4096, 4096),
]


@pytest.fixture
def tuner():
    return FlyDSLGemmTuner()


@pytest.mark.parametrize("m,n,k", SPCONV_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_correctness(tuner, m, n, k, dtype):
    """Verify FlyDSLGemmTuner.matmul matches torch.mm."""
    a = torch.randn(m, k, dtype=dtype, device="cuda")
    b = torch.randn(n, k, dtype=dtype, device="cuda")
    c = torch.zeros(m, n, dtype=dtype, device="cuda")

    tuner.matmul(c, a, b)
    ref = torch.mm(a, b.t())

    rel_err = (c - ref).abs().max().item() / ref.abs().max().item()
    assert rel_err < 0.01, f"rel_err={rel_err:.6f} for shape ({m},{n},{k}) dtype={dtype}"


@pytest.mark.parametrize("m,n,k", [(256, 256, 1024), (512, 256, 2048)])
def test_matmul_beta_accumulate(tuner, m, n, k):
    """Verify beta=1.0 accumulation (spconv multi-KV-position)."""
    dtype = torch.float16
    a = torch.randn(m, k, dtype=dtype, device="cuda")
    b = torch.randn(n, k, dtype=dtype, device="cuda")
    c_init = torch.randn(m, n, dtype=dtype, device="cuda")
    c = c_init.clone()

    tuner.matmul(c, a, b, beta=1.0)
    ref = c_init + torch.mm(a, b.t())

    rel_err = (c - ref).abs().max().item() / ref.abs().max().item()
    assert rel_err < 0.01, f"beta=1.0 accumulation rel_err={rel_err:.6f}"


def test_tile_selection():
    """Verify tile config selection logic."""
    cfg = _select_tile_config(200, 256, 256)
    assert cfg['TILE_N'] == 128 or cfg['TILE_N'] == 256
    assert 256 % cfg['TILE_N'] == 0

    cfg = _select_tile_config(32, 256, 2048)
    assert cfg['SPLIT_K'] > 1, "Small M + large K should use split-K"
