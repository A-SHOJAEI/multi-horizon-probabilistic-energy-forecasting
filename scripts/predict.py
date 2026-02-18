"""Run inference with a trained model on new data."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from energy_forecast.utils.config import load_config
from energy_forecast.data.preprocessing import load_ett_dataset, create_features, scale_data, split_data
from energy_forecast.models.patchtst import PatchTST


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="patchtst", choices=["patchtst", "tft", "nbeats"])
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quantiles = config["quantiles"]

    # Load data
    df = load_ett_dataset(str(project_root / config["data"]["data_dir"]), args.dataset)
    train_df, val_df, test_df = split_data(df)
    train_X, _ = create_features(train_df)
    test_X, _ = create_features(test_df)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_X)
    test_scaled = scaler.transform(test_X)

    # Load model
    seq_len = config["data"]["seq_len"]
    n_features = test_scaled.shape[1]
    ckpt = args.checkpoint or str(project_root / f"checkpoints/{args.model}_{args.dataset}_{args.horizon}.pt")

    model = PatchTST(
        n_features=n_features, seq_len=seq_len, pred_len=args.horizon,
        quantiles=quantiles,
    )
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Predict on last window
    x = torch.tensor(test_scaled[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]  # [pred_len, n_quantiles]

    print(f"\nForecast for next {args.horizon} steps:")
    print(f"{'Step':>5} {'Q10':>8} {'Q25':>8} {'Q50':>8} {'Q75':>8} {'Q90':>8}")
    for i in range(min(20, args.horizon)):
        vals = [f"{pred[i, j]:.4f}" for j in range(len(quantiles))]
        print(f"{i+1:>5} {' '.join(f'{v:>8}' for v in vals)}")
    if args.horizon > 20:
        print(f"  ... ({args.horizon - 20} more steps)")


if __name__ == "__main__":
    main()
