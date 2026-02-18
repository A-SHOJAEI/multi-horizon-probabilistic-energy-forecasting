"""Training loop for time series forecasting models."""

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader


class Trainer:
    """Generic trainer for time series models."""

    def __init__(self, config: dict, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler(enabled=config.get("training", {}).get("amp", True))

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        epochs: int = 100,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        patience: int = 10,
        model_name: str = "model",
    ) -> Dict:
        """Train a model with early stopping."""
        model = model.to(self.device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            n_batches = 0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                with autocast("cuda", enabled=self.config.get("training", {}).get("amp", True)):
                    pred = model(x)
                    loss = criterion(pred, y)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(n_batches, 1)
            scheduler.step()

            # Validate
            val_loss = self._validate(model, val_loader, criterion)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        return {"history": history, "best_val_loss": best_val_loss}

    @torch.no_grad()
    def _validate(self, model, loader, criterion) -> float:
        model.eval()
        total_loss = 0.0
        n = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            with autocast("cuda", enabled=self.config.get("training", {}).get("amp", True)):
                pred = model(x)
                loss = criterion(pred, y)
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def predict(self, model, loader) -> tuple:
        """Generate predictions on a dataset."""
        model.eval()
        model.to(self.device)
        all_preds = []
        all_targets = []

        for x, y in loader:
            x = x.to(self.device)
            with autocast("cuda", enabled=self.config.get("training", {}).get("amp", True)):
                pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

        return np.concatenate(all_preds), np.concatenate(all_targets)
