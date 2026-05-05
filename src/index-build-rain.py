"""
Build population-weighted CHIRPS rainfall index for CCRIF countries.

Inputs:
    data/raw/chirps/
    data/raw/worldpop/
    data/raw/natural-earth/ne_10m_admin_0_countries.zip

Outputs:
    data/interim/rain-index/
        by_country/
            worldpop_chirps_weights_<ISO3>_<year>.csv
            chirps_monthly_metrics_popweighted_<ISO3>_<start>_<end>.csv

        chirps_monthly_metrics_popweighted_all_countries_<start>_<end>.csv
        failed_rain_index_builds.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray as rxr
import regionmask


# ============================================================
# SETTINGS
# ============================================================

COUNTRIES = {
    "AIA": "Anguilla",
    "ATG": "Antigua and Barbuda",
    "BHS": "Bahamas",
    "BRB": "Barbados",
    "BLZ": "Belize",
    "BMU": "Bermuda",
    "VGB": "British Virgin Islands",
    "CYM": "Cayman Islands",
    "DMA": "Dominica",
    "DOM": "Dominican Republic",
    "GRD": "Grenada",
    "HTI": "Haiti",
    "JAM": "Jamaica",
    "MSR": "Montserrat",
    "KNA": "Saint Kitts and Nevis",
    "LCA": "Saint Lucia",
    "VCT": "Saint Vincent and the Grenadines",
    "SXM": "Sint Maarten",
    "TTO": "Trinidad and Tobago",
    "TCA": "Turks and Caicos Islands",
}

START_YEAR = 2007
END_YEAR = 2024
MIN_WORLDPOP_YEAR = 2015

CHIRPS_DIR = Path("data/raw/chirps")
WORLDPOP_DIR = Path("data/raw/worldpop")
NATURAL_EARTH_PATH = Path("data/raw/natural-earth/ne_10m_admin_0_countries.zip")

OUT_DIR = Path("data/interim/rain-index")
COUNTRY_OUT_DIR = OUT_DIR / "by_country"

OUT_DIR.mkdir(parents=True, exist_ok=True)
COUNTRY_OUT_DIR.mkdir(parents=True, exist_ok=True)

RAIN_INDEX_COLUMNS = [
    "monthly_total_mm_pop",
    "monthly_mean_daily_mm_pop",
    "monthly_max_1d_mm_pop",
    "monthly_max_3d_mm_pop",
    "monthly_max_5d_mm_pop",
    "monthly_p95_daily_mm_pop",
    "monthly_p99_daily_mm_pop",
    "monthly_n_days_above_20mm_pop",
    "monthly_n_days_above_50mm_pop",
]


# ============================================================
# FILE NAMING
# ============================================================

def chirps_file_path(year: int) -> Path:
    return CHIRPS_DIR / f"chirps-v2.0.{year}.days_p05.nc"


def worldpop_file_path(iso3: str, year: int) -> Path:
    filename = f"{iso3.lower()}_pop_{year}_CN_1km_R2025A_UA_v1.tif"
    return WORLDPOP_DIR / iso3 / filename


def weights_file_path(iso3: str, start_year: int, end_year: int) -> Path:
    return COUNTRY_OUT_DIR / (
        f"worldpop_chirps_weights_{iso3}_{start_year}_{end_year}.csv"
    )


def monthly_index_file_path(iso3: str, start_year: int, end_year: int) -> Path:
    return COUNTRY_OUT_DIR / (
        f"chirps_monthly_rain_indices_popweighted_{iso3}_{start_year}_{end_year}.csv"
    )


def all_countries_index_file_path(start_year: int, end_year: int) -> Path:
    return OUT_DIR / (
        f"chirps_monthly_rain_indices_popweighted_all_countries_"
        f"{start_year}_{end_year}.csv"
    )


# ============================================================
# LOADERS
# ============================================================

def load_country_polygons() -> gpd.GeoDataFrame:
    if not NATURAL_EARTH_PATH.exists():
        raise FileNotFoundError(
            f"Natural Earth file not found: {NATURAL_EARTH_PATH}. "
            "Download ne_10m_admin_0_countries.zip into data/raw/natural-earth/"
        )

    world = gpd.read_file(NATURAL_EARTH_PATH).copy()

    name_col = "NAME" if "NAME" in world.columns else "ADMIN"
    iso_col = "ADM0_A3" if "ADM0_A3" in world.columns else "ISO_A3"

    world = world[[name_col, iso_col, "geometry"]].copy()
    world = world.rename(columns={name_col: "country", iso_col: "iso_a3"})

    rename_map = {
        "Dominican Rep.": "Dominican Republic",
        "St. Kitts and Nevis": "Saint Kitts and Nevis",
        "St. Lucia": "Saint Lucia",
        "St. Vin. and Gren.": "Saint Vincent and the Grenadines",
        "Antigua and Barb.": "Antigua and Barbuda",
    }

    world["country"] = world["country"].replace(rename_map)
    world = world[world.geometry.notna()].copy()

    if world.crs is None:
        world = world.set_crs("EPSG:4326")
    elif world.crs.to_string() != "EPSG:4326":
        world = world.to_crs("EPSG:4326")

    return world[["country", "iso_a3", "geometry"]].reset_index(drop=True)


def get_country_polygon(
    countries_gdf: gpd.GeoDataFrame,
    iso3: str,
    country_name: str,
) -> gpd.GeoDataFrame:
    country = countries_gdf[countries_gdf["iso_a3"] == iso3].copy()

    if country.empty:
        country = countries_gdf[countries_gdf["country"] == country_name].copy()

    if country.empty:
        raise ValueError(
            f"Country not found in Natural Earth: {country_name} ({iso3}). "
            "Check Natural Earth naming or add a manual override."
        )

    country["country"] = country_name
    country["iso_a3"] = iso3

    return country.reset_index(drop=True)


def load_chirps_precip(year: int) -> xr.DataArray:
    path = chirps_file_path(year)

    if not path.exists():
        raise FileNotFoundError(f"CHIRPS file not found: {path}")

    ds = xr.open_dataset(path)

    if "precip" not in ds:
        raise ValueError(f"'precip' variable not found in CHIRPS file: {path}")

    da = ds["precip"]

    rename_map = {}
    if "latitude" in da.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in da.dims:
        rename_map["longitude"] = "lon"

    if rename_map:
        da = da.rename(rename_map)

    required_dims = {"time", "lat", "lon"}
    missing_dims = required_dims - set(da.dims)
    if missing_dims:
        raise ValueError(f"CHIRPS file missing expected dimensions: {missing_dims}")

    return da


def load_worldpop_raster(iso3: str, year: int):
    path = worldpop_file_path(iso3, year)

    if not path.exists():
        raise FileNotFoundError(f"WorldPop raster not found: {path}")

    pop = rxr.open_rasterio(path, masked=True).squeeze()

    if pop.rio.crs is None:
        raise ValueError(f"WorldPop raster has no CRS: {path}")

    if str(pop.rio.crs) != "EPSG:4326":
        pop = pop.rio.reproject("EPSG:4326")

    return pop


# ============================================================
# CHIRPS GRID / COUNTRY MASK
# ============================================================

def subset_chirps_to_country(
    da: xr.DataArray,
    country_gdf: gpd.GeoDataFrame,
    pad: float = 1.0,
) -> xr.DataArray:
    minx, miny, maxx, maxy = country_gdf.total_bounds
    lat_descending = bool(da["lat"].values[0] > da["lat"].values[-1])

    return da.sel(
        lon=slice(minx - pad, maxx + pad),
        lat=slice(maxy + pad, miny - pad)
        if lat_descending
        else slice(miny - pad, maxy + pad),
    )


def build_country_mask(
    da_sub: xr.DataArray,
    country_gdf: gpd.GeoDataFrame,
) -> xr.DataArray:
    mask = regionmask.mask_geopandas(country_gdf, da_sub.lon, da_sub.lat)
    country_mask = xr.where(mask == 0, 1.0, 0.0)

    selected_cells = int((country_mask > 0).sum().item())
    if selected_cells == 0:
        raise ValueError("No CHIRPS cells selected for country")

    return country_mask


def infer_cell_edges(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 1 or len(coords) < 2:
        raise ValueError("Need at least two coordinate centers to infer edges")

    mids = (coords[:-1] + coords[1:]) / 2.0
    first_edge = coords[0] - (mids[0] - coords[0])
    last_edge = coords[-1] + (coords[-1] - mids[-1])

    return np.concatenate([[first_edge], mids, [last_edge]])


def build_chirps_cell_polygons(
    da_sub: xr.DataArray,
    country_mask: xr.DataArray,
    iso3: str,
    country_name: str,
    year: int,
) -> gpd.GeoDataFrame:
    from shapely.geometry import box

    lats = da_sub["lat"].values
    lons = da_sub["lon"].values

    lat_edges = infer_cell_edges(lats)
    lon_edges = infer_cell_edges(lons)

    mask_vals = country_mask.values
    rows = []

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            if mask_vals[i, j] <= 0:
                continue

            lat0 = min(lat_edges[i], lat_edges[i + 1])
            lat1 = max(lat_edges[i], lat_edges[i + 1])
            lon0 = min(lon_edges[j], lon_edges[j + 1])
            lon1 = max(lon_edges[j], lon_edges[j + 1])

            rows.append(
                {
                    "iso3": iso3,
                    "country": country_name,
                    "year": year,
                    "lat": float(lat),
                    "lon": float(lon),
                    "geometry": box(lon0, lat0, lon1, lat1),
                }
            )

    if not rows:
        raise ValueError("No CHIRPS cell polygons were built")

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


# ============================================================
# POPULATION WEIGHTS
# ============================================================

def population_sum_within_polygon(pop_da, polygon) -> float:
    from shapely.geometry import mapping

    clipped = pop_da.rio.clip(
        [mapping(polygon)],
        crs="EPSG:4326",
        drop=True,
    )

    vals = clipped.values

    if np.ma.isMaskedArray(vals):
        arr = vals.filled(np.nan)
    else:
        arr = vals.astype(float)

    return float(np.nansum(arr))


def aggregate_population_to_cells(
    chirps_cells_gdf: gpd.GeoDataFrame,
    pop_da,
) -> pd.DataFrame:
    rows = []

    for _, row in chirps_cells_gdf.iterrows():
        pop_count = population_sum_within_polygon(pop_da, row.geometry)

        rows.append(
            {
                "iso3": row["iso3"],
                "country": row["country"],
                "year": row["year"],
                "lat": row["lat"],
                "lon": row["lon"],
                "pop_count": pop_count,
            }
        )

    df = pd.DataFrame(rows)

    total_pop = df["pop_count"].sum()
    if total_pop <= 0:
        raise ValueError("Total aggregated population is non-positive")

    df["pop_share"] = df["pop_count"] / total_pop

    return df


def build_population_weights_for_year(
    iso3: str,
    country_name: str,
    year: int,
    da_sub: xr.DataArray,
    country_mask: xr.DataArray,
    pop_da,
) -> pd.DataFrame:
    chirps_cells = build_chirps_cell_polygons(
        da_sub=da_sub,
        country_mask=country_mask,
        iso3=iso3,
        country_name=country_name,
        year=year,
    )

    weights = aggregate_population_to_cells(
        chirps_cells_gdf=chirps_cells,
        pop_da=pop_da,
    )


    return weights


# ============================================================
# RAINFALL INDEX
# ============================================================

def build_daily_popweighted_rainfall(
    da_sub: xr.DataArray,
    country_mask: xr.DataArray,
    weights: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    rain_df = da_sub.to_dataframe(name="precip_mm").reset_index()
    mask_df = country_mask.to_dataframe(name="inside_country").reset_index()

    rain_df = rain_df.merge(
        mask_df,
        on=["lat", "lon"],
        how="left",
    )

    rain_df = rain_df[rain_df["inside_country"] > 0].copy()

    rain_df = rain_df.merge(
        weights[["lat", "lon", "pop_share"]],
        on=["lat", "lon"],
        how="left",
    )

    missing = rain_df[rain_df["pop_share"].isna()][["lat", "lon"]].drop_duplicates()
    if len(missing) > 0:
        print(
            f"Warning: missing pop_share for {len(missing)} CHIRPS cells in {year}. "
            "Assigning zero and renormalizing."
        )

    rain_df["pop_share"] = rain_df["pop_share"].fillna(0.0)

    weight_sum = (
        rain_df[["lat", "lon", "pop_share"]]
        .drop_duplicates()["pop_share"]
        .sum()
    )

    if weight_sum <= 0:
        raise ValueError(f"Population weights sum to zero in {year}")

    rain_df["pop_share"] = rain_df["pop_share"] / weight_sum

    rain_df["time"] = pd.to_datetime(rain_df["time"])
    rain_df["weighted_precip"] = rain_df["precip_mm"] * rain_df["pop_share"]

    daily = (
        rain_df.groupby("time", as_index=False)
        .agg(pop_weighted_precip_mm=("weighted_precip", "sum"))
        .sort_values("time")
        .reset_index(drop=True)
    )

    return daily


def aggregate_daily_to_monthly(
    daily: pd.DataFrame,
    iso3: str,
    country_name: str,
) -> pd.DataFrame:
    """
    Aggregate daily population-weighted rainfall into several
    candidate monthly rainfall indices.

    Notes:
        - Rolling 3-day and 5-day totals are computed over the daily
          series before monthly aggregation.
        - This allows storm windows to cross month boundaries, which is
          often desirable for disaster exposure measurement.
    """
    daily = daily.copy()

    daily["time"] = pd.to_datetime(daily["time"])
    daily = daily.sort_values("time").reset_index(drop=True)

    rain_col = "pop_weighted_precip_mm"

    daily["rolling_3d_mm_pop"] = (
        daily[rain_col]
        .rolling(window=3, min_periods=1)
        .sum()
    )

    daily["rolling_5d_mm_pop"] = (
        daily[rain_col]
        .rolling(window=5, min_periods=1)
        .sum()
    )

    daily["year"] = daily["time"].dt.year
    daily["month"] = daily["time"].dt.month
    daily["year_month"] = daily["time"].dt.to_period("M").astype(str)

    monthly = (
        daily.groupby(["year", "month", "year_month"], as_index=False)
        .agg(
            monthly_total_mm_pop=(rain_col, "sum"),
            monthly_mean_daily_mm_pop=(rain_col, "mean"),
            monthly_max_1d_mm_pop=(rain_col, "max"),
            monthly_max_3d_mm_pop=("rolling_3d_mm_pop", "max"),
            monthly_max_5d_mm_pop=("rolling_5d_mm_pop", "max"),
            monthly_p95_daily_mm_pop=(rain_col, lambda x: x.quantile(0.95)),
            monthly_p99_daily_mm_pop=(rain_col, lambda x: x.quantile(0.99)),
            monthly_n_days_above_20mm_pop=(rain_col, lambda x: int((x >= 20).sum())),
            monthly_n_days_above_50mm_pop=(rain_col, lambda x: int((x >= 50).sum())),
            n_days_observed=(rain_col, "count"),
        )
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )

    monthly.insert(0, "country", country_name)
    monthly.insert(0, "iso3", iso3)
    monthly["date"] = pd.to_datetime(monthly["year_month"])

    ordered_cols = [
        "iso3",
        "country",
        "year",
        "month",
        "year_month",
        "date",
        "monthly_total_mm_pop",
        "monthly_mean_daily_mm_pop",
        "monthly_max_1d_mm_pop",
        "monthly_max_3d_mm_pop",
        "monthly_max_5d_mm_pop",
        "monthly_p95_daily_mm_pop",
        "monthly_p99_daily_mm_pop",
        "monthly_n_days_above_20mm_pop",
        "monthly_n_days_above_50mm_pop",
        "n_days_observed",
    ]

    return monthly[ordered_cols]

def build_country_year_index(
    iso3: str,
    country_name: str,
    year: int,
    country_gdf: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    da = load_chirps_precip(year)
    da_sub = subset_chirps_to_country(da, country_gdf, pad=1.0)
    country_mask = build_country_mask(da_sub, country_gdf)

    population_weight_year = max(year, MIN_WORLDPOP_YEAR)
    pop_da = load_worldpop_raster(iso3, population_weight_year)

    weights = build_population_weights_for_year(
        iso3=iso3,
        country_name=country_name,
        year=year,
        da_sub=da_sub,
        country_mask=country_mask,
        pop_da=pop_da,
    )
    weights["rainfall_year"] = year
    weights["population_weight_year"] = population_weight_year

    daily = build_daily_popweighted_rainfall(
        da_sub=da_sub,
        country_mask=country_mask,
        weights=weights,
        year=year,
    )

    monthly = aggregate_daily_to_monthly(
        daily=daily,
        iso3=iso3,
        country_name=country_name,
    )

    return monthly, weights


# ============================================================
# COUNTRY PIPELINE
# ============================================================

def build_country_index(
    iso3: str,
    country_name: str,
    countries_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    country_gdf = get_country_polygon(
        countries_gdf=countries_gdf,
        iso3=iso3,
        country_name=country_name,
    )

    all_monthly = []
    all_weights = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Processing {country_name} ({iso3}), {year}...")

        monthly_year, weights_year = build_country_year_index(
            iso3=iso3,
            country_name=country_name,
            year=year,
            country_gdf=country_gdf,
        )

        all_monthly.append(monthly_year)
        all_weights.append(weights_year)

    monthly = pd.concat(all_monthly, ignore_index=True)
    weights = pd.concat(all_weights, ignore_index=True)

    monthly_out = monthly_index_file_path(
        iso3=iso3,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )

    weights_out = weights_file_path(
        iso3=iso3,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )

    monthly.to_csv(monthly_out, index=False)
    weights.to_csv(weights_out, index=False)

    print(f"Saved country rainfall index: {monthly_out}")
    print(f"Saved country population weights: {weights_out}")

    return monthly


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading Natural Earth countries...")
    countries_gdf = load_country_polygons()
    print(f"Loaded {len(countries_gdf):,} Natural Earth geometries")

    all_country_panels = []
    failed = []

    for iso3, country_name in COUNTRIES.items():
        try:
            monthly = build_country_index(
            iso3=iso3,
            country_name=country_name,
            countries_gdf=countries_gdf,
            )

            all_country_panels.append(monthly)

        except Exception as e:
            print(f"FAILED: {country_name} ({iso3}) -> {e}")
            failed.append(
                {
                    "iso3": iso3,
                    "country": country_name,
                    "error": str(e),
                }
            )

    if all_country_panels:
        panel = pd.concat(all_country_panels, ignore_index=True)

        out_path = all_countries_index_file_path(
            start_year=START_YEAR,
            end_year=END_YEAR,
        )

        panel.to_csv(out_path, index=False)
        print(f"Saved all-country rainfall index: {out_path}")

    if failed:
        failed_df = pd.DataFrame(failed)
        failed_path = OUT_DIR / "failed_rain_index_builds.csv"
        failed_df.to_csv(failed_path, index=False)
        print(f"Saved failure log: {failed_path}")


if __name__ == "__main__":
    main()