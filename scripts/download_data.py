"""Download ETT datasets."""

from pathlib import Path
import requests
from tqdm import tqdm


ETT_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    print("Downloading ETT datasets...")
    for name, url in ETT_URLS.items():
        dest = data_dir / f"{name}.csv"
        if dest.exists():
            print(f"  {name}: already exists")
            continue

        print(f"  Downloading {name}...")
        response = requests.get(url)
        response.raise_for_status()
        dest.write_text(response.text)
        print(f"  {name}: saved ({len(response.text) // 1024} KB)")

    print("\nAll ETT datasets ready!")
    for f in sorted(data_dir.glob("*.csv")):
        import pandas as pd
        df = pd.read_csv(f)
        print(f"  {f.name}: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
