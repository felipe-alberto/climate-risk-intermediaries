from pathlib import Path
import requests
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask
import matplotlib.pyplot as plt

COUNTRY_NAME = "Trinidad and Tobago"
START_YEAR = 2017
END_YEAR = 2024


OUT_DIR = Path("data/interim")
OUT_DIR.mkdir(parents=True, exist_ok=True)


DATA_DIR = Path("data/raw/chirps")
FILE_NAMES = {
    year: f"chirps-v2.0.{year}.days_p05.nc"
    for year in range(2017, 2025)
}


def build_country_month_metrics_popweighted(
    country_name: str,
    year: int,
    country_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    # 1. Load CHIRPS and subset exactly as before
    if country_gdf.crs is None:
        raise ValueError("country_gdf has no CRS defined")
    if country_gdf.crs.to_string() != "EPSG:4326":
        country_gdf = country_gdf.to_crs("EPSG:4326")

    local_nc = DATA_DIR / FILE_NAMES[year]
    if not local_nc.exists():
        raise FileNotFoundError(f"CHIRPS file not found: {local_nc}")

    ds = xr.open_dataset(local_nc)
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
    lat_descending = bool(da["lat"].values[0] > da["lat"].values[-1])

    da_sub = da.sel(
        lon=slice(minx - pad, maxx + pad),
        lat=slice(maxy + pad, miny - pad) if lat_descending else slice(miny - pad, maxy + pad),
    )

    # 2. Keep only CHIRPS cells in the country
    mask = regionmask.mask_geopandas(country_gdf, da_sub.lon, da_sub.lat)
    country_mask = xr.where(mask == 0, 1.0, 0.0)

    selected_cells = int((country_mask > 0).sum().item())
    if selected_cells == 0:
        raise ValueError(f"No CHIRPS cells selected for {country_name} in {year}")

    # 3. Convert CHIRPS to long dataframe
    rain_df = da_sub.to_dataframe(name="precip_mm").reset_index()

    # Merge only cells inside the country
    mask_df = country_mask.to_dataframe(name="inside_country").reset_index()
    rain_df = rain_df.merge(mask_df, on=["lat", "lon"], how="left")
    rain_df = rain_df[rain_df["inside_country"] > 0].copy()

    # 4. Load population weights
    weights_file = OUT_DIR / f"worldpop_chirps_weights_{country_name.replace(' ', '_')}_{year}.csv"
    if not weights_file.exists():
        raise FileNotFoundError(f"Population weights file not found: {weights_file}")

    weights = pd.read_csv(weights_file)

    # 5. Merge pop shares onto rainfall cells
    rain_df = rain_df.merge(
        weights[["lat", "lon", "pop_share"]],
        on=["lat", "lon"],
        how="left"
    )

    
    missing = rain_df[rain_df["pop_share"].isna()][["lat", "lon"]].drop_duplicates()
    if len(missing) > 0:
        print(
            f"Warning: missing pop_share for some CHIRPS cells in {year}: "
            f"{len(missing)} cells. Assigning 0 for now."
        )
    rain_df["pop_share"] = rain_df["pop_share"].fillna(0.0)
    weight_sum = rain_df["pop_share"].sum()
    if weight_sum <= 0:
        raise ValueError(f"Population shares sum to zero in {year} after filling missing values.")

    rain_df["pop_share"] = rain_df["pop_share"] / weight_sum

    # 6. Compute pop-weighted daily rainfall
    rain_df["time"] = pd.to_datetime(rain_df["time"])
    rain_df["weighted_precip"] = rain_df["precip_mm"] * rain_df["pop_share"]

    daily = (
        rain_df.groupby("time", as_index=False)
        .agg(pop_weighted_precip_mm=("weighted_precip", "sum"))
        .sort_values("time")
        .reset_index(drop=True)
    )

    # 7. Monthly totals
    daily["year"] = daily["time"].dt.year
    daily["month"] = daily["time"].dt.month
    daily["year_month"] = daily["time"].dt.to_period("M").astype(str)

    monthly = (
        daily.groupby(["year", "month", "year_month"], as_index=False)
        .agg(monthly_total_mm_pop=("pop_weighted_precip_mm", "sum"))
    )

    monthly.insert(0, "country", country_name)
    return monthly

# Load country polygon once
world = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
).to_crs("EPSG:4326")

country = world[world["NAME"] == COUNTRY_NAME].copy().reset_index(drop=True)
if country.empty:
    raise ValueError(f"Country not found: {COUNTRY_NAME}")


all_monthly_pop = []

for year in range(START_YEAR, END_YEAR + 1):
    print(f"Processing pop-weighted rainfall for {COUNTRY_NAME}, {year}...")
    monthly_pop_year = build_country_month_metrics_popweighted(COUNTRY_NAME, year, country)
    all_monthly_pop.append(monthly_pop_year)

monthly_pop = pd.concat(all_monthly_pop, ignore_index=True)
monthly_pop["date"] = pd.to_datetime(monthly_pop["year_month"])

outfile = OUT_DIR / f"chirps_monthly_metrics_popweighted_{COUNTRY_NAME.replace(' ', '_')}_{START_YEAR}_{END_YEAR}.csv"
monthly_pop.to_csv(outfile, index=False)
print(f"Saved: {outfile}")