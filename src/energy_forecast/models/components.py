"""Simple N-BEATS and TFT wrappers using direct PyTorch implementations."""

import torch
import torch.nn as nn
import math


class NBeatsBlock(nn.Module):
    """Single N-BEATS block."""

    def __init__(self, input_size: int, theta_size: int, hidden_size: int = 256, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.fc = nn.Sequential(*layers)
        self.theta_b = nn.Linear(hidden_size, theta_size)
        self.theta_f = nn.Linear(hidden_size, theta_size)

    def forward(self, x):
        h = self.fc(x)
        return self.theta_b(h), self.theta_f(h)


class NBeats(nn.Module):
    """N-BEATS: Neural Basis Expansion Analysis for Time Series.

    Simplified generic architecture implementation.
    """

    def __init__(
        self,
        seq_len: int = 336,
        pred_len: int = 96,
        hidden_size: int = 256,
        n_blocks: int = 6,
        quantiles: list = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.quantiles = quantiles or [0.5]
        self.n_quantiles = len(self.quantiles)

        self.blocks = nn.ModuleList([
            NBeatsBlock(seq_len, seq_len + pred_len, hidden_size)
            for _ in range(n_blocks)
        ])

        self.quantile_head = nn.Linear(pred_len, pred_len * self.n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, n_features] — uses last feature (target)
        Returns:
            [B, pred_len, n_quantiles] or [B, pred_len]
        """
        # Use target channel
        residual = x[:, :, -1]  # [B, seq_len]
        forecast = torch.zeros(x.shape[0], self.pred_len, device=x.device)

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast[:, :self.seq_len]
            forecast = forecast + block_forecast[:, :self.pred_len]

        # Quantile outputs
        out = self.quantile_head(forecast)  # [B, pred_len * n_quantiles]
        out = out.view(-1, self.pred_len, self.n_quantiles)

        if self.n_quantiles == 1:
            return out.squeeze(-1)
        return out


class SimpleTFT(nn.Module):
    """Simplified Temporal Fusion Transformer.

    Uses variable selection, LSTM encoder, and multi-head attention.
    """

    def __init__(
        self,
        n_features: int = 7,
        seq_len: int = 336,
        pred_len: int = 96,
        hidden_size: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        quantiles: list = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.quantiles = quantiles or [0.5]
        self.n_quantiles = len(self.quantiles)

        # Variable selection
        self.var_select = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_features),
            nn.Softmax(dim=-1),
        )

        # Input projection
        self.input_proj = nn.Linear(n_features, hidden_size)

        # LSTM encoder
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=2,
            batch_first=True, dropout=dropout,
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_size, n_heads, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, pred_len * self.n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, n_features]
        Returns:
            [B, pred_len, n_quantiles]
        """
        B = x.shape[0]

        # Variable selection
        weights = self.var_select(x)  # [B, seq_len, n_features]
        x_selected = x * weights

        # Input projection
        h = self.input_proj(x_selected)  # [B, seq_len, hidden]

        # LSTM
        lstm_out, _ = self.lstm(h)  # [B, seq_len, hidden]

        # Self-attention on LSTM output
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        h = self.attn_norm(lstm_out + attn_out)

        # Use last hidden state for prediction
        out = self.output_proj(h[:, -1, :])  # [B, pred_len * n_quantiles]
        out = out.view(B, self.pred_len, self.n_quantiles)

        if self.n_quantiles == 1:
            return out.squeeze(-1)
        return out
