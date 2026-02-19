"""Train all models on all datasets and horizons."""

import json
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from energy_forecast.utils.config import load_config
from energy_forecast.data.preprocessing import load_ett_dataset, split_data, create_features, scale_data
from energy_forecast.data.loader import create_dataloaders
from energy_forecast.models.patchtst import PatchTST, QuantileLoss
from energy_forecast.models.components import NBeats, SimpleTFT
from energy_forecast.models.baselines import run_baseline_evaluation
from energy_forecast.training.trainer import Trainer
from energy_forecast.evaluation.metrics import compute_all_metrics, save_results


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    config = load_config()
    set_seed(config["training"]["seed"])

    quantiles = config["quantiles"]
    all_results = {}
    total_start = time.time()

    for dataset_name in config["data"]["datasets"]:
        print(f"\n{'#'*70}")
        print(f"# Dataset: {dataset_name}")
        print(f"{'#'*70}")

        # Load and prepare data
        df = load_ett_dataset(str(project_root / config["data"]["data_dir"]), dataset_name)
        train_df, val_df, test_df = split_data(df, config["data"]["train_ratio"], config["data"]["val_ratio"])

        train_X, train_y = create_features(train_df, config["data"]["target_col"])
        val_X, val_y = create_features(val_df, config["data"]["target_col"])
        test_X, test_y = create_features(test_df, config["data"]["target_col"])

        train_scaled, val_scaled, test_scaled, scaler = scale_data(train_X, val_X, test_X)
        n_features = train_scaled.shape[1]

        for pred_len in config["data"]["pred_horizons"]:
            print(f"\n{'='*60}")
            print(f"Horizon: {pred_len} steps")
            print(f"{'='*60}")

            seq_len = config["data"]["seq_len"]

            # Create data loaders
            train_loader, val_loader, test_loader = create_dataloaders(
                train_scaled, val_scaled, test_scaled,
                seq_len=seq_len, pred_len=pred_len,
                batch_size=config["patchtst"]["batch_size"],
                num_workers=config["training"]["num_workers"],
            )

            trainer = Trainer(config)

            # 1. PatchTST
            print(f"\n--- PatchTST ---")
            patchtst = PatchTST(
                n_features=n_features,
                seq_len=seq_len,
                pred_len=pred_len,
                patch_len=config["patchtst"]["patch_len"],
                stride=config["patchtst"]["stride"],
                d_model=config["patchtst"]["d_model"],
                n_heads=config["patchtst"]["n_heads"],
                n_layers=config["patchtst"]["n_layers"],
                d_ff=config["patchtst"]["d_ff"],
                dropout=config["patchtst"]["dropout"],
                quantiles=quantiles,
            )

            criterion = QuantileLoss(quantiles)
            result = trainer.train(
                patchtst, train_loader, val_loader, criterion,
                epochs=config["patchtst"]["epochs"],
                lr=config["patchtst"]["lr"],
                patience=config["training"]["patience"],
                model_name="patchtst",
            )

            preds, targets = trainer.predict(patchtst, test_loader)
            metrics = compute_all_metrics(preds, targets, quantiles)
            metrics["training_history"] = result["history"]
            key = f"patchtst/{dataset_name}/{pred_len}"
            all_results[key] = metrics
            print(f"  PatchTST: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")

            # Save checkpoint
            ckpt_dir = Path(config["output"]["checkpoint_dir"])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(patchtst.state_dict(), ckpt_dir / f"patchtst_{dataset_name}_{pred_len}.pt")
            del patchtst
            torch.cuda.empty_cache()

            # 2. SimpleTFT
            print(f"\n--- TFT ---")
            tft = SimpleTFT(
                n_features=n_features,
                seq_len=seq_len,
                pred_len=pred_len,
                hidden_size=config["tft"]["hidden_size"],
                n_heads=config["tft"]["attention_head_size"],
                dropout=config["tft"]["dropout"],
                quantiles=quantiles,
            )

            result = trainer.train(
                tft, train_loader, val_loader, criterion,
                epochs=config["tft"]["epochs"],
                lr=config["tft"]["lr"],
                patience=config["training"]["patience"],
                model_name="tft",
            )

            preds, targets = trainer.predict(tft, test_loader)
            metrics = compute_all_metrics(preds, targets, quantiles)
            metrics["training_history"] = result["history"]
            key = f"tft/{dataset_name}/{pred_len}"
            all_results[key] = metrics
            print(f"  TFT: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")

            torch.save(tft.state_dict(), ckpt_dir / f"tft_{dataset_name}_{pred_len}.pt")
            del tft
            torch.cuda.empty_cache()

            # 3. N-BEATS
            print(f"\n--- N-BEATS ---")
            nbeats = NBeats(
                seq_len=seq_len,
                pred_len=pred_len,
                hidden_size=config["nbeats"]["widths"][0],
                n_blocks=sum(config["nbeats"]["num_blocks"]),
                quantiles=quantiles,
            )

            result = trainer.train(
                nbeats, train_loader, val_loader, criterion,
                epochs=config["nbeats"]["epochs"],
                lr=config["nbeats"]["lr"],
                patience=config["training"]["patience"],
                model_name="nbeats",
            )

            preds, targets = trainer.predict(nbeats, test_loader)
            metrics = compute_all_metrics(preds, targets, quantiles)
            metrics["training_history"] = result["history"]
            key = f"nbeats/{dataset_name}/{pred_len}"
            all_results[key] = metrics
            print(f"  N-BEATS: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")

            torch.save(nbeats.state_dict(), ckpt_dir / f"nbeats_{dataset_name}_{pred_len}.pt")
            del nbeats
            torch.cuda.empty_cache()

            # 4. Statistical baselines
            print(f"\n--- Baselines ---")
            baseline_results = run_baseline_evaluation(
                train_y, test_scaled, seq_len, pred_len, target_idx=-1
            )
            for bname, bmetrics in baseline_results.items():
                key = f"{bname}/{dataset_name}/{pred_len}"
                all_results[key] = bmetrics
                print(f"  {bname}: MSE={bmetrics['mse']:.4f}, MAE={bmetrics['mae']:.4f}")

    elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"All training completed in {elapsed/3600:.1f} hours")
    print(f"{'='*70}")

    # Save results
    results_dir = Path(config["output"]["results_dir"])
    save_results(all_results, str(results_dir / "all_results.json"))

    # Print summary table
    print(f"\n{'Model':<20} {'Dataset':<10} {'Horizon':<10} {'MSE':>8} {'MAE':>8}")
    print("-" * 60)
    for key in sorted(all_results.keys()):
        parts = key.split("/")
        model = parts[0]
        dataset = parts[1] if len(parts) > 1 else ""
        horizon = parts[2] if len(parts) > 2 else ""
        r = all_results[key]
        print(f"{model:<20} {dataset:<10} {horizon:<10} {r['mse']:>8.4f} {r['mae']:>8.4f}")


if __name__ == "__main__":
    main()
