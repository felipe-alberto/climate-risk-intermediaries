from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask
import rioxarray as rxr

from shapely.geometry import box, mapping
from rasterio.features import geometry_mask


COUNTRY_NAME = "Trinidad and Tobago"
ISO3 = "TTO"
START_YEAR = 2017
END_YEAR = 2024

CHIRPS_DIR = Path("data/raw/chirps")
WORLDPOP_DIR = Path("data/raw/worldpop")
OUT_DIR = Path("data/interim")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHIRPS_FILE_NAMES = {
    year: f"chirps-v2.0.{year}.days_p05.nc"
    for year in range(2017, 2025)
}

WORLDPOP_FILE_NAMES = {
    year: f"{ISO3.lower()}_pop_{year}_CN_1km_R2025A_UA_v1.tif"
    for year in range(2017, 2025)
}


def load_country_polygon(country_name: str) -> gpd.GeoDataFrame:
    world = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
    ).to_crs("EPSG:4326")

    country = world[world["NAME"] == country_name].copy().reset_index(drop=True)
    if country.empty:
        raise ValueError(f"Country not found: {country_name}")

    if country.crs is None:
        raise ValueError("Country GeoDataFrame has no CRS")
    if country.crs.to_string() != "EPSG:4326":
        country = country.to_crs("EPSG:4326")

    return country


def load_chirps_precip(year: int) -> xr.DataArray:
    if year not in CHIRPS_FILE_NAMES:
        raise ValueError(f"Year {year} not configured for CHIRPS")

    local_nc = CHIRPS_DIR / CHIRPS_FILE_NAMES[year]
    if not local_nc.exists():
        raise FileNotFoundError(f"CHIRPS file not found: {local_nc}")

    try:
        ds = xr.open_dataset(local_nc)
    except Exception as e:
        raise OSError(f"Could not open CHIRPS NetCDF for year {year}: {local_nc}") from e

    if "precip" not in ds:
        raise ValueError(f"'precip' variable not found in CHIRPS file for {year}")

    da = ds["precip"]

    rename_map = {}
    if "latitude" in da.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in da.dims:
        rename_map["longitude"] = "lon"
    if rename_map:
        da = da.rename(rename_map)

    return da


def subset_chirps_to_country(da: xr.DataArray, country_gdf: gpd.GeoDataFrame, pad: float = 1.0) -> xr.DataArray:
    minx, miny, maxx, maxy = country_gdf.total_bounds
    lat_descending = bool(da["lat"].values[0] > da["lat"].values[-1])

    da_sub = da.sel(
        lon=slice(minx - pad, maxx + pad),
        lat=slice(maxy + pad, miny - pad) if lat_descending else slice(miny - pad, maxy + pad),
    )

    return da_sub


def build_country_mask(da_sub: xr.DataArray, country_gdf: gpd.GeoDataFrame) -> xr.DataArray:
    mask = regionmask.mask_geopandas(country_gdf, da_sub.lon, da_sub.lat)
    country_mask = xr.where(mask == 0, 1.0, 0.0)

    selected_cells = int((country_mask > 0).sum().item())
    if selected_cells == 0:
        raise ValueError("No CHIRPS cells selected for country")

    return country_mask


def infer_cell_edges(coords: np.ndarray) -> np.ndarray:
    """
    Given 1D coordinate centers, infer cell edges.
    Works for ascending or descending coordinates.
    """
    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 1 or len(coords) < 2:
        raise ValueError("Need at least two coordinate centers to infer edges")

    mids = (coords[:-1] + coords[1:]) / 2.0
    first_edge = coords[0] - (mids[0] - coords[0])
    last_edge = coords[-1] + (coords[-1] - mids[-1])

    edges = np.concatenate([[first_edge], mids, [last_edge]])
    return edges


def build_chirps_cell_polygons(
    da_sub: xr.DataArray,
    country_mask: xr.DataArray,
    country_name: str,
    year: int,
) -> gpd.GeoDataFrame:
    lats = da_sub["lat"].values
    lons = da_sub["lon"].values

    lat_edges = infer_cell_edges(lats)
    lon_edges = infer_cell_edges(lons)

    rows = []

    # country_mask uses the same lat/lon grid as da_sub
    mask_vals = country_mask.values

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            if mask_vals[i, j] <= 0:
                continue

            lat0 = min(lat_edges[i], lat_edges[i + 1])
            lat1 = max(lat_edges[i], lat_edges[i + 1])
            lon0 = min(lon_edges[j], lon_edges[j + 1])
            lon1 = max(lon_edges[j], lon_edges[j + 1])

            geom = box(lon0, lat0, lon1, lat1)

            rows.append(
                {
                    "country": country_name,
                    "year": year,
                    "lat": float(lat),
                    "lon": float(lon),
                    "geometry": geom,
                }
            )

    if not rows:
        raise ValueError("No CHIRPS cell polygons were built")

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return gdf


def load_worldpop_raster(year: int):
    if year not in WORLDPOP_FILE_NAMES:
        raise ValueError(f"Year {year} not configured for WorldPop")

    tif_path = WORLDPOP_DIR / WORLDPOP_FILE_NAMES[year]
    if not tif_path.exists():
        raise FileNotFoundError(f"WorldPop TIFF not found: {tif_path}")

    pop = rxr.open_rasterio(tif_path, masked=True).squeeze()

    if pop.rio.crs is None:
        raise ValueError(f"WorldPop raster has no CRS: {tif_path}")

    # Reproject to EPSG:4326 if needed
    if str(pop.rio.crs) != "EPSG:4326":
        pop = pop.rio.reproject("EPSG:4326")

    return pop


def population_sum_within_polygon(pop_da, polygon) -> float:
    """
    Sum WorldPop values whose pixel centers fall inside the polygon mask.
    This is a simple first-pass approach.
    """
    # Clip to polygon bbox first for speed
    clipped = pop_da.rio.clip([mapping(polygon)], crs="EPSG:4326", drop=True)

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

    for idx, row in chirps_cells_gdf.iterrows():
        pop_count = population_sum_within_polygon(pop_da, row.geometry)

        rows.append(
            {
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


def save_weights(df: pd.DataFrame, country_name: str, year: int) -> Path:
    out_path = OUT_DIR / f"worldpop_chirps_weights_{country_name.replace(' ', '_')}_{year}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def main():
    country = load_country_polygon(COUNTRY_NAME)

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\nProcessing {COUNTRY_NAME}, {year}...")

        da = load_chirps_precip(year)
        da_sub = subset_chirps_to_country(da, country, pad=1.0)
        country_mask = build_country_mask(da_sub, country)

        chirps_cells = build_chirps_cell_polygons(
            da_sub=da_sub,
            country_mask=country_mask,
            country_name=COUNTRY_NAME,
            year=year,
        )
        print(f"{year}: CHIRPS cells in country = {len(chirps_cells)}")

        pop_da = load_worldpop_raster(year)
        weights = aggregate_population_to_cells(chirps_cells, pop_da)

        # Sanity check
        print(
            f"{year}: total_pop = {weights['pop_count'].sum():,.2f}, "
            f"share_sum = {weights['pop_share'].sum():.6f}"
        )

        out_path = save_weights(weights, COUNTRY_NAME, year)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()