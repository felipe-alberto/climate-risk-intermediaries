from pathlib import Path
import requests

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-"
    "climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)

OUT_DIR = Path("data/raw/ibtracs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IBTRACS_FILE = OUT_DIR / "ibtracs_ALL_list_v04r01.csv"


def download_file(url: str, dest: Path, overwrite: bool = False) -> None:
    if dest.exists() and not overwrite:
        print(f"File already exists: {dest}")
        return

    print(f"Downloading: {url}")

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()

        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"Saved: {dest} ({size_mb:.2f} MB)")


def main():
    download_file(IBTRACS_URL, IBTRACS_FILE)


if __name__ == "__main__":
    main()