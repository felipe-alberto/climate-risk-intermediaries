from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1) USER INPUTS
# ============================================================

# Path to the downloaded IBTrACS CSV file
IBTRACS_CSV = Path("data/raw/ibtracs/ibtracs.ALL.list.v04r01.csv")

# Simple country reference points for a first pass.
# Later, replace with polygons / centroids from Natural Earth or GADM.
COUNTRY_POINTS = {
    "Trinidad and Tobago": (10.6918, -61.2225),
    "Jamaica": (18.1096, -77.2975),
    "Barbados": (13.1939, -59.5432),
    "Belize": (17.1899, -88.4976),
    "Dominica": (15.4150, -61.3710),
    "Grenada": (12.1165, -61.6790),
    "St. Lucia": (13.9094, -60.9789),
    "St. Vincent and the Grenadines": (13.2528, -61.1971),
    "Antigua and Barbuda": (17.0608, -61.7964),
    "St. Kitts and Nevis": (17.3578, -62.7830),
    "The Bahamas": (25.0343, -77.3963),
}

# Distance bands to summarize storm proximity
DISTANCE_BANDS_KM = [100, 200, 300]

# ============================================================
# 2) HELPERS
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance between a point and arrays of points.
    Inputs in degrees. Output in kilometers.
    """
    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(np.asarray(lat2))
    lon2 = np.radians(np.asarray(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def saffir_simpson_category(wind_kt):
    """
    Approximate Saffir-Simpson category from sustained wind in knots.
    Returns:
      0 = tropical storm / below hurricane
      1..5 = hurricane category
    """
    if pd.isna(wind_kt):
        return np.nan

    if wind_kt < 64:
        return 0
    elif wind_kt < 83:
        return 1
    elif wind_kt < 96:
        return 2
    elif wind_kt < 113:
        return 3
    elif wind_kt < 137:
        return 4
    else:
        return 5


def infer_time_step_hours(times):
    """
    Infer the modal time step in hours for one storm track.
    IBTrACS often uses 3-hour or 6-hour spacing depending on source/segment.
    """
    times = pd.Series(times).sort_values().dropna()
    if len(times) < 2:
        return np.nan

    diffs = times.diff().dropna().dt.total_seconds() / 3600.0
    if len(diffs) == 0:
        return np.nan

    # modal-ish robust choice
    return float(diffs.round(3).mode().iloc[0])


# ============================================================
# 3) LOAD IBTrACS
# ============================================================

# IBTrACS CSVs usually have a descriptive second row after the header.
# In many NOAA examples, users skip row 1.
df = pd.read_csv(IBTRACS_CSV, skiprows=[1], low_memory=False)

# Keep only rows with coordinates and time
needed_cols = ["SID", "NAME", "SEASON", "BASIN", "ISO_TIME", "LAT", "LON"]
for col in needed_cols:
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' not found in IBTrACS file.")

df = df.copy()
df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
df["LON"] = pd.to_numeric(df["LON"], errors="coerce")

# Prefer WMO wind if available, otherwise USA wind
wind_col = None
for candidate in ["WMO_WIND", "USA_WIND", "TOKYO_WIND", "CMA_WIND", "HKO_WIND"]:
    if candidate in df.columns:
        wind_col = candidate
        break

if wind_col is None:
    raise ValueError("No wind column found (e.g. WMO_WIND or USA_WIND).")

df[wind_col] = pd.to_numeric(df[wind_col], errors="coerce")

# Drop rows with missing essentials
df = df.dropna(subset=["SID", "ISO_TIME", "LAT", "LON"]).copy()

# Optional: restrict to North Atlantic for Caribbean work
# NA = North Atlantic basin in IBTrACS
df = df[df["BASIN"].astype(str).str.upper() == "NA"].copy()

# ============================================================
# 4) BUILD COUNTRY-STORM TABLE
# ============================================================

rows = []

for sid, g in df.groupby("SID", sort=False):
    g = g.sort_values("ISO_TIME").copy()

    storm_name = g["NAME"].dropna().astype(str).replace(" ", np.nan).dropna()
    storm_name = storm_name.iloc[0] if len(storm_name) else None

    season = g["SEASON"].dropna().iloc[0] if g["SEASON"].notna().any() else None
    basin = g["BASIN"].dropna().iloc[0] if g["BASIN"].notna().any() else None
    start_time = g["ISO_TIME"].min()
    end_time = g["ISO_TIME"].max()
    step_hours = infer_time_step_hours(g["ISO_TIME"])

    lats = g["LAT"].to_numpy()
    lons = g["LON"].to_numpy()
    winds = g[wind_col].to_numpy()
    times = g["ISO_TIME"].to_numpy()

    for country, (clat, clon) in COUNTRY_POINTS.items():
        distances = haversine_km(clat, clon, lats, lons)

        min_idx = int(np.nanargmin(distances))
        min_distance_km = float(distances[min_idx])
        closest_time = pd.Timestamp(times[min_idx])

        row = {
            "country": country,
            "storm_id": sid,
            "storm_name": storm_name,
            "season": season,
            "basin": basin,
            "start_time": start_time,
            "end_time": end_time,
            "closest_approach_time": closest_time,
            "min_distance_km": min_distance_km,
            "max_wind_kt_near_country": np.nan,
            "storm_category_max_near_country": np.nan,
            "track_time_step_hours": step_hours,
            "landfall_like_dummy": int(min_distance_km <= 50),  # crude first-pass proxy
        }

        # Summaries within distance bands
        for band in DISTANCE_BANDS_KM:
            mask = distances <= band
            row[f"hours_within_{band}km"] = (
                float(mask.sum() * step_hours) if pd.notna(step_hours) else np.nan
            )

        # Wind near country: use max wind within 300 km, else fallback to closest point wind
        near_mask = distances <= 300
        if near_mask.any() and np.isfinite(winds[near_mask]).any():
            local_max_wind = np.nanmax(winds[near_mask])
        else:
            local_max_wind = winds[min_idx] if np.isfinite(winds[min_idx]) else np.nan

        row["max_wind_kt_near_country"] = float(local_max_wind) if pd.notna(local_max_wind) else np.nan
        row["storm_category_max_near_country"] = (
            saffir_simpson_category(local_max_wind)
            if pd.notna(local_max_wind)
            else np.nan
        )

        rows.append(row)

country_storm = pd.DataFrame(rows)

# ============================================================
# 5) FILTER TO PLAUSIBLE EXPOSURES
# ============================================================

# Keep only storms that came within 500 km of the country
country_storm = country_storm[country_storm["min_distance_km"] <= 500].copy()

# Sort for readability
country_storm = country_storm.sort_values(
    ["country", "season", "start_time", "min_distance_km"]
).reset_index(drop=True)

# ============================================================
# 6) SAVE
# ============================================================

outdir = Path("data/processed/tc")
outdir.mkdir(parents=True, exist_ok=True)

outfile = outdir / "ibtracs_country_storm_exposure.csv"
country_storm.to_csv(outfile, index=False)

print(f"Saved: {outfile}")
print(country_storm.head(20).to_string(index=False))