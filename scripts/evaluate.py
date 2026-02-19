"""Generate evaluation plots from training results."""

import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from energy_forecast.evaluation.analysis import plot_model_comparison, plot_training_curves


def main():
    results_dir = project_root / "results"
    results_file = results_dir / "all_results.json"

    if not results_file.exists():
        print("No results found. Run scripts/train.py first.")
        sys.exit(1)

    with open(results_file) as f:
        all_results = json.load(f)

    print("Generating evaluation plots...")

    # Model comparison charts
    for metric in ["mse", "mae"]:
        plot_model_comparison(
            all_results, metric,
            f"Model Comparison — {metric.upper()}",
            str(results_dir / f"model_comparison_{metric}.png"),
        )

    # Training curves for each model/dataset/horizon
    for key, result in all_results.items():
        if "training_history" in result:
            safe_key = key.replace("/", "_")
            plot_training_curves(
                result["training_history"],
                f"Training Curves: {key}",
                str(results_dir / f"training_curves_{safe_key}.png"),
            )

    print(f"All plots saved to {results_dir}/")


if __name__ == "__main__":
    main()
