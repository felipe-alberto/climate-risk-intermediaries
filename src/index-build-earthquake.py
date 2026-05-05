"""
Build reduced-form earthquake exposure index for CCRIF countries.

Inputs:
    data/raw/usgs-earthquakes/usgs_earthquakes_2007_2024.csv
    data/raw/natural-earth/ne_110m_admin_0_countries.zip

Outputs:
    data/interim/earthquake-index/
        by_country/
            eq_country_event_pairs_<ISO3>_<start>_<end>.csv
            eq_country_month_panel_<ISO3>_<start>_<end>.csv

        eq_country_month_panel_all_countries_<start>_<end>.csv
        failed_earthquake_index_builds.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


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
    "DMA": "Dominica",
    "LCA": "Saint Lucia",
    "NIC": "Nicaragua",
    "VGB": "British Virgin Islands",
}

START_YEAR = 2007
END_YEAR = 2024
MAX_DIST_KM = 500

RAW_EQ_FILE = Path("data/raw/usgs-earthquakes/usgs_earthquakes_2007_2024.csv")
NATURAL_EARTH_FILE = Path("data/raw/natural-earth/ne_10m_admin_0_countries.zip")

OUT_DIR = Path("data/interim/earthquake-index")
COUNTRY_OUT_DIR = OUT_DIR / "by_country"

OUT_DIR.mkdir(parents=True, exist_ok=True)
COUNTRY_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILE NAMING
# ============================================================

def event_pairs_file_path(iso3: str) -> Path:
    return COUNTRY_OUT_DIR / f"eq_country_event_pairs_{iso3}_{START_YEAR}_{END_YEAR}.csv"


def monthly_index_file_path(iso3: str) -> Path:
    return COUNTRY_OUT_DIR / f"eq_country_month_panel_{iso3}_{START_YEAR}_{END_YEAR}.csv"


def all_countries_index_file_path() -> Path:
    return OUT_DIR / f"eq_country_month_panel_all_countries_{START_YEAR}_{END_YEAR}.csv"


# ============================================================
# LOADERS
# ============================================================

def load_countries() -> gpd.GeoDataFrame:
    if not NATURAL_EARTH_FILE.exists():
        raise FileNotFoundError(f"Natural Earth file not found: {NATURAL_EARTH_FILE}")

    world = gpd.read_file(NATURAL_EARTH_FILE).copy()

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


def load_earthquakes() -> gpd.GeoDataFrame:
    if not RAW_EQ_FILE.exists():
        raise FileNotFoundError(
            f"USGS earthquake file not found: {RAW_EQ_FILE}. "
            "Run the USGS download script first."
        )

    df = pd.read_csv(RAW_EQ_FILE)

    required = ["id", "time", "latitude", "longitude", "mag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Earthquake file missing columns: {missing}")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time", "latitude", "longitude", "mag"]).copy()

    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["year_month"] = df["time"].dt.to_period("M").astype(str)

    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )


# ============================================================
# EARTHQUAKE-COUNTRY MATCHING
# ============================================================

def build_country_event_pairs(
    eq_gdf: gpd.GeoDataFrame,
    country_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Build earthquake-country pairs for one country.

    Keeps earthquakes whose epicenter is within MAX_DIST_KM of the country
    polygon. Then computes a reduced-form distance-decayed shaking proxy:

        shake_proxy = 10^(0.5 * magnitude) * exp(-distance_km / 200)
    """
    eq_m = eq_gdf.to_crs(epsg=3857)
    country_m = country_gdf.to_crs(epsg=3857).copy()

    country_buffered = country_m.copy()
    country_buffered["geometry"] = country_buffered.geometry.buffer(MAX_DIST_KM * 1000)

    pairs = gpd.sjoin(
        eq_m,
        country_buffered[["country", "iso_a3", "geometry"]],
        how="inner",
        predicate="intersects",
    ).drop(columns=["index_right"])

    if pairs.empty:
        return pd.DataFrame(
            columns=[
                "id",
                "time",
                "year",
                "month",
                "year_month",
                "mag",
                "place",
                "country",
                "iso_a3",
                "dist_km",
                "shake_proxy",
            ]
        )

    actual_geom = country_m[["country", "geometry"]].rename(
        columns={"geometry": "country_geom"}
    )

    pairs = pairs.merge(actual_geom, on="country", how="left")

    pairs["dist_km"] = pairs.geometry.distance(pairs["country_geom"]) / 1000.0
    pairs["dist_km"] = pairs["dist_km"].fillna(0.0)

    pairs["shake_proxy"] = (
        (10 ** (0.5 * pairs["mag"]))
        * np.exp(-pairs["dist_km"] / 200.0)
    )

    keep_cols = [
        "id",
        "time",
        "year",
        "month",
        "year_month",
        "mag",
        "place",
        "country",
        "iso_a3",
        "dist_km",
        "shake_proxy",
    ]
    keep_cols = [c for c in keep_cols if c in pairs.columns]

    pairs = pairs[keep_cols].copy()
    pairs = pairs.drop_duplicates(subset=["id", "country"]).copy()
    pairs = pairs.sort_values(["country", "year", "month", "time"]).reset_index(drop=True)

    return pairs


# ============================================================
# MONTHLY EARTHQUAKE INDEX
# ============================================================

def collapse_to_country_month_index(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(
            columns=[
                "country",
                "iso_a3",
                "year",
                "month",
                "n_eq_nearby",
                "monthly_max_mag_nearby",
                "monthly_sum_shake_proxy",
                "monthly_max_shake_proxy",
                "monthly_min_dist_km",
            ]
        )

    return (
        pairs.groupby(["country", "iso_a3", "year", "month"], as_index=False)
        .agg(
            n_eq_nearby=("id", "count"),
            monthly_max_mag_nearby=("mag", "max"),
            monthly_sum_shake_proxy=("shake_proxy", "sum"),
            monthly_max_shake_proxy=("shake_proxy", "max"),
            monthly_min_dist_km=("dist_km", "min"),
        )
        .sort_values(["country", "year", "month"])
        .reset_index(drop=True)
    )


def make_balanced_country_month_panel(
    panel: pd.DataFrame,
    country_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    country_name = country_gdf["country"].iloc[0]
    iso3 = country_gdf["iso_a3"].iloc[0]

    months = pd.period_range(f"{START_YEAR}-01", f"{END_YEAR}-12", freq="M")

    full = pd.DataFrame({"date": months.to_timestamp()})
    full["country"] = country_name
    full["iso3"] = iso3
    full["iso_a3"] = iso3
    full["year"] = full["date"].dt.year
    full["month"] = full["date"].dt.month
    full["year_month"] = full["date"].dt.to_period("M").astype(str)

    out = full.merge(
        panel,
        on=["country", "iso_a3", "year", "month"],
        how="left",
    )

    out["n_eq_nearby"] = out["n_eq_nearby"].fillna(0).astype(int)
    out["monthly_sum_shake_proxy"] = out["monthly_sum_shake_proxy"].fillna(0.0)
    out["monthly_max_shake_proxy"] = out["monthly_max_shake_proxy"].fillna(0.0)

    ordered_cols = [
        "iso3",
        "country",
        "year",
        "month",
        "year_month",
        "date",
        "n_eq_nearby",
        "monthly_max_mag_nearby",
        "monthly_sum_shake_proxy",
        "monthly_max_shake_proxy",
        "monthly_min_dist_km",
    ]

    return out[ordered_cols].sort_values(["country", "year", "month"]).reset_index(drop=True)


# ============================================================
# COUNTRY PIPELINE
# ============================================================

def build_country_earthquake_index(
    eq_gdf: gpd.GeoDataFrame,
    country_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    country_name = country_gdf["country"].iloc[0]
    iso3 = country_gdf["iso_a3"].iloc[0]

    print(f"\nProcessing {country_name} ({iso3})...")

    pairs = build_country_event_pairs(eq_gdf, country_gdf)
    print(f"  Event-country pairs: {len(pairs):,}")

    monthly = collapse_to_country_month_index(pairs)
    balanced = make_balanced_country_month_panel(monthly, country_gdf)

    pairs_out = event_pairs_file_path(iso3)
    panel_out = monthly_index_file_path(iso3)

    pairs.to_csv(pairs_out, index=False)
    balanced.to_csv(panel_out, index=False)

    print(f"  Saved event-pair audit file: {pairs_out}")
    print(f"  Saved monthly earthquake index: {panel_out}")

    return balanced


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading Natural Earth countries...")
    countries_gdf = load_countries()
    print(f"Loaded {len(countries_gdf):,} Natural Earth geometries")

    print("Loading earthquakes...")
    earthquakes = load_earthquakes()
    print(f"Loaded {len(earthquakes):,} earthquakes")

    all_panels = []
    failed = []

    for iso3, country_name in COUNTRIES.items():
        try:
            country_gdf = get_country_polygon(
                countries_gdf=countries_gdf,
                iso3=iso3,
                country_name=country_name,
            )

            panel = build_country_earthquake_index(
                eq_gdf=earthquakes,
                country_gdf=country_gdf,
            )

            all_panels.append(panel)

        except Exception as e:
            print(f"FAILED: {country_name} ({iso3}) -> {e}")
            failed.append(
                {
                    "iso3": iso3,
                    "country": country_name,
                    "error": str(e),
                }
            )

    if all_panels:
        all_panel = pd.concat(all_panels, ignore_index=True)

        all_out = all_countries_index_file_path()
        all_panel.to_csv(all_out, index=False)

        print(f"\nSaved combined monthly earthquake index: {all_out}")
        print("\nSample:")
        print(all_panel.head(20).to_string(index=False))

    if failed:
        failed_df = pd.DataFrame(failed)
        failed_path = OUT_DIR / "failed_earthquake_index_builds.csv"
        failed_df.to_csv(failed_path, index=False)

        print(f"\nSaved failure log: {failed_path}")


if __name__ == "__main__":
    main()