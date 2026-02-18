"""Basic tests for forecasting models."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from energy_forecast.models.patchtst import PatchTST, QuantileLoss
from energy_forecast.models.components import NBeats, SimpleTFT


def test_patchtst_forward():
    model = PatchTST(n_features=7, seq_len=336, pred_len=96, quantiles=[0.1, 0.5, 0.9])
    model.eval()
    x = torch.randn(4, 336, 7)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 96, 3)


def test_nbeats_forward():
    model = NBeats(seq_len=336, pred_len=96, quantiles=[0.1, 0.5, 0.9])
    model.eval()
    x = torch.randn(4, 336, 7)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 96, 3)


def test_tft_forward():
    model = SimpleTFT(n_features=7, seq_len=336, pred_len=96, quantiles=[0.1, 0.5, 0.9])
    model.eval()
    x = torch.randn(4, 336, 7)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 96, 3)


def test_quantile_loss():
    criterion = QuantileLoss([0.1, 0.5, 0.9])
    preds = torch.randn(8, 96, 3)
    targets = torch.randn(8, 96)
    loss = criterion(preds, targets)
    assert loss.shape == ()
    assert loss.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
