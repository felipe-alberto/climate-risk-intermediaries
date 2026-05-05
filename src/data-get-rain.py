from pathlib import Path
import time
import requests


START_YEAR = 2007
END_YEAR = 2024

DATA_DIR = Path("data/raw/chirps")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
    "global_daily/netcdf/p05"
)


def chirps_url(year: int) -> str:
    return f"{BASE_URL}/chirps-v2.0.{year}.days_p05.nc"


def chirps_path(year: int) -> Path:
    return DATA_DIR / f"chirps-v2.0.{year}.days_p05.nc"


def download_chirps_year(
    year: int,
    overwrite: bool = False,
    max_retries: int = 3,
    timeout: int = 120,
    chunk_size: int = 1024 * 1024,
) -> Path:
    url = chirps_url(year)
    local_nc = chirps_path(year)
    tmp_path = local_nc.with_suffix(local_nc.suffix + ".part")

    if local_nc.exists() and not overwrite:
        print(f"[skip] {year}: {local_nc}")
        return local_nc

    if tmp_path.exists():
        tmp_path.unlink()

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[download] {year}: attempt {attempt}/{max_retries}")

            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()

                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

            if tmp_path.stat().st_size == 0:
                raise RuntimeError(f"Downloaded empty file for {year}")

            tmp_path.replace(local_nc)
            print(f"[saved] {year}: {local_nc}")
            return local_nc

        except Exception as e:
            print(f"[error] {year}: {e}")

            if tmp_path.exists():
                tmp_path.unlink()

            if attempt < max_retries:
                time.sleep(5)
            else:
                raise RuntimeError(f"Failed to download CHIRPS file for {year}") from e


def main():
    downloaded_files = []

    for year in range(START_YEAR, END_YEAR + 1):
        path = download_chirps_year(year)
        downloaded_files.append(path)

    print("\nDone.")
    print(f"Downloaded or verified {len(downloaded_files)} CHIRPS files.")
    print(f"Directory: {DATA_DIR}")


if __name__ == "__main__":
    main()