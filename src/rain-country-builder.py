
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask
import matplotlib.pyplot as plt

DATA_DIR = Path("data/raw/chirps")
FILE_NAMES = {
    year: f"chirps-v2.0.{year}.days_p05.nc"
    for year in range(2017, 2025)
}

def build_country_month_metrics(country_name: str, year: int, country_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    
    
    # Checks for GDF
    if country_gdf.crs is None:
        raise ValueError("country_gdf has no CRS defined")
    
    if country_gdf.crs.to_string() != "EPSG:4326":
        country_gdf = country_gdf.to_crs("EPSG:4326")

    if year not in FILE_NAMES:
        raise ValueError(f"Year {year} not configured in FILE_NAMES")

    local_nc = DATA_DIR / FILE_NAMES[year]
    if not local_nc.exists():
        raise FileNotFoundError(f"CHIRPS file not found: {local_nc}")
    
    try:
        ds = xr.open_dataset(local_nc)
    except Exception as e:
        raise OSError(f"Could not open NetCDF for year {year}: {local_nc}") from e
    if "precip" not in ds:
        raise ValueError(f"'precip' variable not found for {year}")

    da = ds["precip"]

    rename_map = {}
    if "latitude" in da.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in da.dims:
        rename_map["longitude"] = "lon"
    if rename_map:
        da = da.rename(rename_map)

    minx, miny, maxx, maxy = country_gdf.total_bounds
    pad = 1.0
    lat_descending = bool(da.lat[0] > da.lat[-1])

    da_sub = da.sel(
        lon=slice(minx - pad, maxx + pad),
        lat=slice(maxy + pad, miny - pad) if lat_descending else slice(miny - pad, maxy + pad),
    )

    mask = regionmask.mask_geopandas(country_gdf, da_sub.lon, da_sub.lat)
    country_mask = xr.where(mask == 0, 1.0, 0.0)

    selected_cells = int((country_mask > 0).sum().item())
    print(f"{year}: selected grid cells = {selected_cells}")
    if selected_cells == 0:
        raise ValueError(f"No CHIRPS cells selected for {country_name} in {year}")

    lat_weights = np.cos(np.deg2rad(da_sub["lat"]))
    weights = (country_mask * lat_weights).fillna(0.0)

    country_daily = da_sub.weighted(weights).mean(dim=("lat", "lon")).load()

    daily = country_daily.to_dataframe(name="precip_mm").reset_index()
    daily["time"] = pd.to_datetime(daily["time"])
    daily = daily.dropna(subset=["precip_mm"]).sort_values("time").reset_index(drop=True)

    daily["rain_1d_mm"] = daily["precip_mm"]
    daily["rain_3d_mm"] = daily["precip_mm"].rolling(window=3, min_periods=3).sum()
    daily["rain_5d_mm"] = daily["precip_mm"].rolling(window=5, min_periods=5).sum()

    daily["year"] = daily["time"].dt.year
    daily["month"] = daily["time"].dt.month
    daily["year_month"] = daily["time"].dt.to_period("M").astype(str)

    monthly = (
        daily.groupby(["year", "month", "year_month"], as_index=False)
        .agg(
            monthly_total_mm=("precip_mm", "sum"),
            max_1d_mm=("rain_1d_mm", "max"),
            max_3d_mm=("rain_3d_mm", "max"),
            max_5d_mm=("rain_5d_mm", "max"),
        )
    )

    monthly.insert(0, "country", country_name)
    return monthly

# Main Run

COUNTRY_NAME = "Trinidad and Tobago"
START_YEAR = 2017
END_YEAR = 2024

OUT_DIR = Path("data/interim")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load country polygon once
world = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
).to_crs("EPSG:4326")

country = world[world["NAME"] == COUNTRY_NAME].copy().reset_index(drop=True)
if country.empty:
    raise ValueError(f"Country not found: {COUNTRY_NAME}")

# Build multi-year monthly rainfall panel
all_monthly = []

for year in range(START_YEAR, END_YEAR + 1):
    print(f"Processing {COUNTRY_NAME}, {year}...")
    monthly_year = build_country_month_metrics(COUNTRY_NAME, year, country)
    all_monthly.append(monthly_year)

    # optional: save one interim file per year
    yearly_outfile = OUT_DIR / f"chirps_monthly_metrics_{COUNTRY_NAME.replace(' ', '_')}_{year}.csv"
    monthly_year.to_csv(yearly_outfile, index=False)
    print(f"Saved yearly interim file: {yearly_outfile}")

# Combine all years
monthly = pd.concat(all_monthly, ignore_index=True)
monthly["date"] = pd.to_datetime(monthly["year_month"])

# Save full panel
panel_outfile = OUT_DIR / (
    f"chirps_monthly_metrics_{COUNTRY_NAME.replace(' ', '_')}_{START_YEAR}_{END_YEAR}.csv"
)
monthly.to_csv(panel_outfile, index=False)
print(f"Saved full panel: {panel_outfile}")