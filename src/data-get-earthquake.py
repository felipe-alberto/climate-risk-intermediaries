from pathlib import Path
from io import StringIO
from calendar import monthrange

import requests
import pandas as pd


RAW_DIR = Path("data/raw/usgs-earthquakes")
RAW_DIR.mkdir(parents=True, exist_ok=True)

EQ_FILE = RAW_DIR / "usgs_earthquakes_2007_2024.csv"

START_YEAR = 2007
END_YEAR = 2024
MIN_MAG = 4.5


def fetch_usgs_catalog_chunk(start_date: str, end_date: str, min_mag: float) -> pd.DataFrame:
    """
    Fetch one chunk from the USGS earthquake catalog.
    """
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    params = {
        "format": "csv",
        "starttime": start_date,
        "endtime": end_date,
        "minmagnitude": min_mag,
        "orderby": "time-asc",
        "eventtype": "earthquake",
        "limit": 20000,
    }

    r = requests.get(base_url, params=params, timeout=240)
    print(f"URL: {r.url}")
    print(f"Status: {r.status_code}")

    if not r.ok:
        print(r.text[:1500])
        r.raise_for_status()

    if not r.text.strip():
        return pd.DataFrame()

    return pd.read_csv(StringIO(r.text))


def download_usgs_catalog_chunked(
    start_year: int,
    end_year: int,
    min_mag: float,
    dest: Path,
    overwrite: bool = False,
) -> None:
    """
    Download USGS earthquake catalog in chunks.

    First tries yearly pulls. If a year exceeds the USGS API cap,
    falls back to month-by-month pulls.
    """
    if dest.exists() and not overwrite:
        print(f"File already exists: {dest}")
        return

    frames = []

    for year in range(start_year, end_year + 1):
        y_start = f"{year}-01-01"
        y_end = f"{year}-12-31"

        print(f"\nTrying yearly pull for {year}...")

        try:
            df_year = fetch_usgs_catalog_chunk(y_start, y_end, min_mag)
            print(f"Loaded {len(df_year):,} rows for {year}")
            frames.append(df_year)

        except requests.HTTPError:
            print(f"Year {year} exceeded limit or failed. Falling back to monthly pulls.")

            for month in range(1, 13):
                last_day = monthrange(year, month)[1]
                m_start = f"{year}-{month:02d}-01"
                m_end = f"{year}-{month:02d}-{last_day:02d}"

                print(f"  Pulling {m_start} to {m_end}...")
                df_month = fetch_usgs_catalog_chunk(m_start, m_end, min_mag)
                print(f"  Loaded {len(df_month):,} rows")
                frames.append(df_month)

    if not frames:
        raise ValueError("No earthquake data was downloaded.")

    df = pd.concat(frames, ignore_index=True)

    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"]).copy()
    else:
        df = df.drop_duplicates().copy()

    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)

    print(f"\nSaved {len(df):,} unique earthquakes to: {dest}")


def main():
    download_usgs_catalog_chunked(
        start_year=START_YEAR,
        end_year=END_YEAR,
        min_mag=MIN_MAG,
        dest=EQ_FILE,
    )


if __name__ == "__main__":
    main()