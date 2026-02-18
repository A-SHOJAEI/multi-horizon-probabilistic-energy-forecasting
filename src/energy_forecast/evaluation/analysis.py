"""Visualization and analysis utilities for time series forecasting."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    quantiles: List[float],
    title: str,
    output_path: str,
    sample_idx: int = 0,
):
    """Plot prediction with uncertainty intervals for a single sample."""
    fig, ax = plt.subplots(figsize=(12, 4))
    pred_len = targets.shape[1]
    x = np.arange(pred_len)

    ax.plot(x, targets[sample_idx], "k-", label="Actual", linewidth=2)

    if predictions.ndim == 3:
        median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
        ax.plot(x, predictions[sample_idx, :, median_idx], "b-", label="Median", linewidth=1.5)

        # 80% interval
        if 0.1 in quantiles and 0.9 in quantiles:
            lo = quantiles.index(0.1)
            hi = quantiles.index(0.9)
            ax.fill_between(
                x, predictions[sample_idx, :, lo], predictions[sample_idx, :, hi],
                alpha=0.2, color="blue", label="80% PI"
            )

        # 50% interval
        if 0.25 in quantiles and 0.75 in quantiles:
            lo = quantiles.index(0.25)
            hi = quantiles.index(0.75)
            ax.fill_between(
                x, predictions[sample_idx, :, lo], predictions[sample_idx, :, hi],
                alpha=0.3, color="blue", label="50% PI"
            )
    else:
        ax.plot(x, predictions[sample_idx], "b-", label="Prediction", linewidth=1.5)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_comparison(
    all_results: Dict,
    metric: str,
    title: str,
    output_path: str,
):
    """Bar chart comparing models across datasets and horizons."""
    models = sorted(set(k.split("/")[0] for k in all_results.keys()))
    datasets = sorted(set(k.split("/")[1] for k in all_results.keys() if "/" in k))
    horizons = sorted(set(k.split("/")[2] for k in all_results.keys() if k.count("/") >= 2))

    if not horizons:
        return

    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 5), squeeze=False)

    for h_idx, horizon in enumerate(horizons):
        ax = axes[0, h_idx]
        x_pos = np.arange(len(datasets))
        width = 0.8 / len(models)

        for m_idx, model in enumerate(models):
            values = []
            for dataset in datasets:
                key = f"{model}/{dataset}/{horizon}"
                if key in all_results and metric in all_results[key]:
                    values.append(all_results[key][metric])
                else:
                    values.append(0)
            ax.bar(x_pos + m_idx * width, values, width, label=model)

        ax.set_xticks(x_pos + width * (len(models) - 1) / 2)
        ax.set_xticklabels(datasets, rotation=45, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Horizon = {horizon}")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_calibration(
    predictions: np.ndarray,
    targets: np.ndarray,
    quantiles: List[float],
    title: str,
    output_path: str,
):
    """Plot calibration curve for quantile forecasts."""
    fig, ax = plt.subplots(figsize=(6, 6))

    observed_freqs = []
    for i, q in enumerate(quantiles):
        freq = np.mean(targets <= predictions[:, :, i])
        observed_freqs.append(freq)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(quantiles, observed_freqs, "bo-", label="Observed")
    ax.set_xlabel("Nominal Quantile")
    ax.set_ylabel("Observed Frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(history: Dict, title: str, output_path: str):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train")
    ax.plot(epochs, history["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
