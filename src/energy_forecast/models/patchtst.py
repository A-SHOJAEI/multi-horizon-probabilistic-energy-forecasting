"""PatchTST: Patch Time Series Transformer for forecasting.

Implements the channel-independent PatchTST model from scratch with
quantile regression heads for probabilistic forecasting.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PatchTST(nn.Module):
    """Patch Time Series Transformer.

    Channel-independent patching approach where each input channel
    is processed independently through shared Transformer layers.
    """

    def __init__(
        self,
        n_features: int = 7,
        seq_len: int = 336,
        pred_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 256,
        dropout: float = 0.2,
        quantiles: list = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.quantiles = quantiles or [0.5]
        self.n_quantiles = len(self.quantiles)

        # Number of patches
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Patch embedding
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=self.n_patches + 1, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction heads (one per quantile for the target channel)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.n_patches * d_model, pred_len * self.n_quantiles),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, seq_len, n_features]

        Returns:
            If quantiles > 1: [B, pred_len, n_quantiles]
            Else: [B, pred_len]
        """
        B = x.shape[0]

        # Use only the target channel (last feature) for channel-independent processing
        x_target = x[:, :, -1]  # [B, seq_len]

        # Create patches
        patches = x_target.unfold(1, self.patch_len, self.stride)  # [B, n_patches, patch_len]

        # Embed patches
        z = self.patch_embed(patches)  # [B, n_patches, d_model]
        z = self.pos_enc(z)

        # Transformer
        z = self.transformer(z)  # [B, n_patches, d_model]

        # Predict
        out = self.head(z)  # [B, pred_len * n_quantiles]
        out = out.view(B, self.pred_len, self.n_quantiles)

        if self.n_quantiles == 1:
            return out.squeeze(-1)
        return out


class QuantileLoss(nn.Module):
    """Quantile regression loss (pinball loss)."""

    def __init__(self, quantiles: list):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, pred_len, n_quantiles]
            targets: [B, pred_len]
        """
        if predictions.dim() == 2:
            predictions = predictions.unsqueeze(-1)

        targets = targets.unsqueeze(-1)  # [B, pred_len, 1]
        errors = targets - predictions  # [B, pred_len, n_quantiles]

        losses = []
        for i, q in enumerate(self.quantiles):
            e = errors[:, :, i]
            loss = torch.max(q * e, (q - 1) * e)
            losses.append(loss.mean())

        return torch.stack(losses).mean()
