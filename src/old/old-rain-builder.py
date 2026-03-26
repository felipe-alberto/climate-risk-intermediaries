# test_chirps_country_month_metrics.py
#
# Prototype:
# 1. Download one CHIRPS daily NetCDF year file
# 2. Load one country polygon
# 3. Compute area-weighted country-average daily rainfall
# 4. Build monthly metrics:
#       - max_1d_mm
#       - max_3d_mm
#       - max_5d_mm
#       - monthly_total_mm
#
# Example target:
#   Trinidad and Tobago, 2024

from pathlib import Path
import requests
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask

# -----------------------------
# User inputs
# -----------------------------
COUNTRY_NAME = "Trinidad and Tobago"
YEAR = 2024

DATA_DIR = Path("data/raw/chirps")
OUT_DIR = Path("data/interim")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1. Download CHIRPS daily file
# -----------------------------
# Public CHIRPS daily NetCDF yearly files live in:
# https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/
chirps_url = (
    f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
    f"global_daily/netcdf/p05/chirps-v2.0.{YEAR}.days_p05.nc"
)

local_nc = DATA_DIR / f"chirps-v2.0.{YEAR}.days_p05.nc"

if not local_nc.exists():
    print(f"Downloading CHIRPS daily file for {YEAR}...")
    r = requests.get(chirps_url, stream=True, timeout=120)
    r.raise_for_status()
    with open(local_nc, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    print(f"Saved: {local_nc}")
else:
    print(f"Using existing file: {local_nc}")

# -----------------------------
# 2. Load country boundaries
# -----------------------------
world = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
).to_crs("EPSG:4326")

country = world[world["NAME"] == COUNTRY_NAME].copy().reset_index(drop=True)
if country.empty:
    raise ValueError(f"Country not found in Natural Earth: {COUNTRY_NAME}")

print(f"Loaded polygon for: {COUNTRY_NAME}")
print("Country bounds:", country.total_bounds)

# -----------------------------
# 3. Load CHIRPS data
# -----------------------------
ds = xr.open_dataset(local_nc)

if "precip" not in ds:
    raise ValueError(f"'precip' variable not found. Variables are: {list(ds.data_vars)}")

da = ds["precip"]

# Standardize names to lat/lon
rename_map = {}
if "latitude" in da.dims:
    rename_map["latitude"] = "lat"
if "longitude" in da.dims:
    rename_map["longitude"] = "lon"
if rename_map:
    da = da.rename(rename_map)

print("Data dims:", da.dims)
print("Lon range:", float(da.lon.min()), float(da.lon.max()))
print("Lat range:", float(da.lat.min()), float(da.lat.max()))
print("First 5 lons:", da.lon.values[:5])
print("First 5 lats:", da.lat.values[:5])

# -----------------------------
# 4. Spatial subset + mask for country
# -----------------------------
minx, miny, maxx, maxy = country.total_bounds
pad = 1.0

lat_descending = bool(da.lat[0] > da.lat[-1])

da_sub = da.sel(
    lon=slice(minx - pad, maxx + pad),
    lat=slice(maxy + pad, miny - pad) if lat_descending else slice(miny - pad, maxy + pad),
)

print("Subset shape:", da_sub.shape)
print("Subset lon range:", float(da_sub.lon.min()), float(da_sub.lon.max()))
print("Subset lat range:", float(da_sub.lat.min()), float(da_sub.lat.max()))

mask = regionmask.mask_geopandas(country, da_sub.lon, da_sub.lat)

# Because we reset_index(drop=True), the single country has region id 0
country_mask = xr.where(mask == 0, 1.0, 0.0)

mask_vals = mask.values[~np.isnan(mask.values)]
print("Mask unique values:", np.unique(mask_vals)[:10] if mask_vals.size else "all nan")
print("Selected grid cells:", int((country_mask > 0).sum().item()))

if int((country_mask > 0).sum().item()) == 0:
    raise SystemExit("Stopping: no CHIRPS grid cells selected for country")

# Latitude weights for area adjustment
lat_weights = np.cos(np.deg2rad(da_sub["lat"]))
lat_weights.name = "lat_weights"

# No missing values allowed in weights
weights = (country_mask * lat_weights).fillna(0.0)

# Compute weighted country-average daily rainfall on subset only
country_daily = da_sub.weighted(weights).mean(dim=("lat", "lon")).load()

# Convert to DataFrame
daily = country_daily.to_dataframe(name="precip_mm").reset_index()
daily["time"] = pd.to_datetime(daily["time"])
daily = daily.dropna(subset=["precip_mm"]).sort_values("time").reset_index(drop=True)

print("\nDaily series preview:")
print(daily.head())
print("Number of daily rows:", len(daily))

if daily.empty:
    raise SystemExit("Stopping: daily rainfall series is empty after aggregation")

# -----------------------------
# 5. Compute rolling rainfall metrics
# -----------------------------
daily["rain_1d_mm"] = daily["precip_mm"]
daily["rain_3d_mm"] = daily["precip_mm"].rolling(window=3, min_periods=3).sum()
daily["rain_5d_mm"] = daily["precip_mm"].rolling(window=5, min_periods=5).sum()

daily["year"] = daily["time"].dt.year
daily["month"] = daily["time"].dt.month
daily["year_month"] = daily["time"].dt.to_period("M").astype(str)

# -----------------------------
# 6. Collapse to country-month metrics
# -----------------------------
monthly = (
    daily.groupby(["year", "month", "year_month"], as_index=False)
    .agg(
        monthly_total_mm=("precip_mm", "sum"),
        max_1d_mm=("rain_1d_mm", "max"),
        max_3d_mm=("rain_3d_mm", "max"),
        max_5d_mm=("rain_5d_mm", "max"),
    )
)

monthly.insert(0, "country", COUNTRY_NAME)

print("\nMonthly rainfall metrics:")
print(monthly)

# -----------------------------
# 7. Save outputs
# -----------------------------
daily_out = OUT_DIR / f"chirps_daily_{COUNTRY_NAME.replace(' ', '_').replace('.', '')}_{YEAR}.csv"
monthly_out = OUT_DIR / f"chirps_monthly_metrics_{COUNTRY_NAME.replace(' ', '_').replace('.', '')}_{YEAR}.csv"

daily.to_csv(daily_out, index=False)
monthly.to_csv(monthly_out, index=False)

print(f"\nSaved daily series:   {daily_out}")
print(f"Saved monthly table:  {monthly_out}")

import matplotlib.pyplot as plt

# -----------------------------
# 7. Plot monthly rainfall metrics
# -----------------------------
monthly["date"] = pd.to_datetime(monthly["year_month"])

plt.figure(figsize=(10, 6))
plt.plot(monthly["date"], monthly["monthly_total_mm"], marker="o", label="Monthly total")
plt.plot(monthly["date"], monthly["max_1d_mm"], marker="o", label="Max 1-day")
plt.plot(monthly["date"], monthly["max_3d_mm"], marker="o", label="Max 3-day")
plt.plot(monthly["date"], monthly["max_5d_mm"], marker="o", label="Max 5-day")

plt.title(f"Rainfall metrics over time: {COUNTRY_NAME} ({YEAR})")
plt.xlabel("Month")
plt.ylabel("Rainfall (mm)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plot_path = OUT_DIR / f"chirps_monthly_metrics_{COUNTRY_NAME.replace(' ', '_').replace('.', '')}_{YEAR}.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved plot: {plot_path}")

# -----------------------------
# 8. Load payout months for this country-year
# -----------------------------
payouts = pd.read_csv("data/raw/payouts.csv")

# Keep the columns we actually need, including Month
payouts = payouts[["Country", "Year", "Month", "Amount (USD)"]].copy()

# Clean variables
payouts["Year"] = pd.to_numeric(payouts["Year"], errors="coerce")
payouts["Month"] = pd.to_numeric(payouts["Month"], errors="coerce")

payouts["Amount (USD)"] = (
    payouts["Amount (USD)"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)
payouts["Amount (USD)"] = pd.to_numeric(payouts["Amount (USD)"], errors="coerce")

payouts = payouts.dropna(subset=["Country", "Year", "Month", "Amount (USD)"]).copy()
payouts["Year"] = payouts["Year"].astype(int)
payouts["Month"] = payouts["Month"].astype(int)

# Filter to Trinidad and Tobago in 2024
country_payouts = payouts[
    (payouts["Country"] == COUNTRY_NAME) &
    (payouts["Year"] == YEAR)
].copy()

print("\nFiltered payouts for country-year:")
print(country_payouts)

# Collapse to one row per month
monthly_payouts = (
    country_payouts
    .groupby(["Year", "Month"], as_index=False)["Amount (USD)"]
    .sum()
    .rename(columns={
        "Year": "year",
        "Month": "month",
        "Amount (USD)": "total_payout_usd"
    })
)

monthly_payouts["date"] = pd.to_datetime(
    monthly_payouts["year"].astype(str)
    + "-"
    + monthly_payouts["month"].astype(str).str.zfill(2)
    + "-01"
)

print("\nMonthly payouts to overlay:")
print(monthly_payouts)
# -----------------------------
# 9. Plot rainfall metrics + payout months
# -----------------------------
plt.figure(figsize=(11, 6))
plt.plot(monthly["date"], monthly["monthly_total_mm"], marker="o", label="Monthly total")
plt.plot(monthly["date"], monthly["max_1d_mm"], marker="o", label="Max 1-day")
plt.plot(monthly["date"], monthly["max_3d_mm"], marker="o", label="Max 3-day")
plt.plot(monthly["date"], monthly["max_5d_mm"], marker="o", label="Max 5-day")

first_line = True
for _, row in monthly_payouts.iterrows():
    plt.axvline(
        row["date"],
        linestyle="--",
        alpha=0.7,
        label="Payout month" if first_line else None
    )
    first_line = False

plt.title(f"Rainfall metrics and payout months: {COUNTRY_NAME} ({YEAR})")
plt.xlabel("Month")
plt.ylabel("Rainfall (mm)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()