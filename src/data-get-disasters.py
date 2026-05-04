"""
data-get-disasters.py

Ingest EM-DAT disaster data into the project raw-data folder.

Baseline use:
    python src/data-get-disasters.py \
        --local-file ~/Downloads/emdat_public.xlsx

Optional direct download:
    python src/data-get-disasters.py \
        --url "https://..." 

Output:
    data/raw/emdat/emdat.csv
"""

from pathlib import Path
import argparse
import shutil
import requests
import pandas as pd


RAW_DIR = Path("data/raw/emdat")
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RAW_DIR / "emdat.csv"


def copy_local_file(local_file: str) -> Path:
    src = Path(local_file).expanduser()

    if not src.exists():
        raise FileNotFoundError(f"Local file not found: {src}")

    dst = RAW_DIR / src.name
    shutil.copy2(src, dst)

    return dst


def download_file(url: str) -> Path:
    suffix = Path(url.split("?")[0]).suffix or ".xlsx"
    dst = RAW_DIR / f"emdat_download{suffix}"

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(dst, "wb") as f:
        f.write(response.content)

    return dst


def convert_to_csv(input_file: Path) -> pd.DataFrame:
    suffix = input_file.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(input_file)
    elif suffix == ".csv":
        df = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    df.to_csv(OUTPUT_CSV, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-file",
        type=str,
        default=None,
        help="Path to manually downloaded EM-DAT .xlsx or .csv file.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Direct URL to EM-DAT export, if available.",
    )

    args = parser.parse_args()

    if args.local_file is None and args.url is None:
        raise ValueError(
            "Provide either --local-file or --url. "
            "EM-DAT usually requires registration/login before download."
        )

    if args.local_file is not None:
        input_file = copy_local_file(args.local_file)
    else:
        input_file = download_file(args.url)

    df = convert_to_csv(input_file)

    print(f"Saved raw EM-DAT CSV to: {OUTPUT_CSV}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")


if __name__ == "__main__":
    main()