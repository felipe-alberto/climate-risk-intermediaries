from pathlib import Path
import time
import requests


IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-"
    "climate-stewardship-ibtracs/v04r01/access/csv/"
    "ibtracs.ALL.list.v04r01.csv"
)

OUT_DIR = Path("data/raw/ibtracs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IBTRACS_FILE = OUT_DIR / "ibtracs_ALL_list_v04r01.csv"


def download_file(
    url: str,
    dest: Path,
    overwrite: bool = False,
    max_retries: int = 3,
    timeout: int = 120,
    chunk_size: int = 1024 * 1024,
) -> Path:
    tmp_path = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and not overwrite:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"[skip] Existing file: {dest} ({size_mb:.2f} MB)")
        return dest

    if tmp_path.exists():
        tmp_path.unlink()

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[download] attempt {attempt}/{max_retries}")
            print(f"URL: {url}")

            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()

                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

            if tmp_path.stat().st_size == 0:
                raise RuntimeError("Downloaded empty IBTrACS file")

            tmp_path.replace(dest)

            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"[saved] {dest} ({size_mb:.2f} MB)")
            return dest

        except Exception as e:
            print(f"[error] {e}")

            if tmp_path.exists():
                tmp_path.unlink()

            if attempt < max_retries:
                time.sleep(5)
            else:
                raise RuntimeError("Failed to download IBTrACS file") from e


def main():
    download_file(IBTRACS_URL, IBTRACS_FILE)


if __name__ == "__main__":
    main()