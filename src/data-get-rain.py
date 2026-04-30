from pathlib import Path
import requests
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask
import matplotlib.pyplot as plt


START_YEAR = 2017
END_YEAR = 2024

DATA_DIR = Path("data/raw/chirps")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_chirps_year(year: int) -> Path:
    url = (
        f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
        f"global_daily/netcdf/p05/chirps-v2.0.{year}.days_p05.nc"
    )
    local_nc = DATA_DIR / f"chirps-v2.0.{year}.days_p05.nc"

    if not local_nc.exists():
        print(f"Downloading CHIRPS daily file for {year}...")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(local_nc, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"Saved: {local_nc}")
    else:
        print(f"Using existing file: {local_nc}")

    return local_nc

for year in range(START_YEAR, END_YEAR + 1):
    download_chirps_year(year)

