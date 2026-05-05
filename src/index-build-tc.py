"""
Build tropical cyclone / wind exposure index for CCRIF countries.

Inputs:
    data/raw/ibtracs/ibtracs_ALL_list_v04r01.csv
    data/raw/natural-earth/ne_10m_admin_0_countries.zip

Outputs:
    data/interim/tc-index/
        by_country/
            tc_country_track_hits_<ISO3>_<start>_<end>.csv
            tc_storm_country_month_<ISO3>_<start>_<end>.csv
            tc_country_month_panel_<ISO3>_<start>_<end>.csv

        tc_country_month_panel_all_countries_<start>_<end>.csv
        failed_tc_index_builds.csv
"""

from pathlib import Path

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
}

START_YEAR = 2007
END_YEAR = 2024
HIT_BUFFER_KM = 50

RAW_IBTRACS_FILE = Path("data/raw/ibtracs/ibtracs_ALL_list_v04r01.csv")
NATURAL_EARTH_FILE = Path("data/raw/natural-earth/ne_10m_admin_0_countries.zip")

OUT_DIR = Path("data/interim/tc-index")
COUNTRY_OUT_DIR = OUT_DIR / "by_country"

OUT_DIR.mkdir(parents=True, exist_ok=True)
COUNTRY_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILE NAMING
# ============================================================

def track_hits_file_path(iso3: str) -> Path:
    return COUNTRY_OUT_DIR / f"tc_country_track_hits_{iso3}_{START_YEAR}_{END_YEAR}.csv"


def storm_month_file_path(iso3: str) -> Path:
    return COUNTRY_OUT_DIR / f"tc_storm_country_month_{iso3}_{START_YEAR}_{END_YEAR}.csv"


def monthly_index_file_path(iso3: str) -> Path:
    return COUNTRY_OUT_DIR / f"tc_country_month_panel_{iso3}_{START_YEAR}_{END_YEAR}.csv"


def all_countries_index_file_path() -> Path:
    return OUT_DIR / f"tc_country_month_panel_all_countries_{START_YEAR}_{END_YEAR}.csv"


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


def add_best_wind_column(df: pd.DataFrame) -> pd.DataFrame:
    wind_candidates = [
        "USA_WIND",
        "WMO_WIND",
        "TOKYO_WIND",
        "CMA_WIND",
        "HKO_WIND",
        "NEWDELHI_WIND",
        "REUNION_WIND",
        "BOM_WIND",
        "NADI_WIND",
        "WELLINGTON_WIND",
        "DS824_WIND",
        "TD9636_WIND",
        "TD9635_WIND",
        "NEUMANN_WIND",
        "MLC_WIND",
    ]

    existing = [c for c in wind_candidates if c in df.columns]

    if not existing:
        raise ValueError("No known wind columns found in IBTrACS file.")

    for col in existing:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["best_wind_kt"] = df[existing].max(axis=1, skipna=True)
    df["best_wind_mps"] = df["best_wind_kt"] * 0.514444

    return df


def load_ibtracs() -> pd.DataFrame:
    if not RAW_IBTRACS_FILE.exists():
        raise FileNotFoundError(
            f"IBTrACS file not found: {RAW_IBTRACS_FILE}. "
            "Run the IBTrACS download script first."
        )

    df = pd.read_csv(RAW_IBTRACS_FILE, skiprows=[1], low_memory=False)

    required = ["SID", "SEASON", "NAME", "ISO_TIME", "LAT", "LON", "NATURE"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"IBTrACS file missing columns: {missing}")

    df["SEASON"] = pd.to_numeric(df["SEASON"], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce", utc=True)
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")

    df = df.dropna(subset=["SEASON", "ISO_TIME", "LAT", "LON"]).copy()
    df["SEASON"] = df["SEASON"].astype(int)

    df = df[(df["SEASON"] >= START_YEAR) & (df["SEASON"] <= END_YEAR)].copy()
    df = df[df["NATURE"].notna()].copy()

    df["NAME"] = df["NAME"].fillna("UNNAMED").astype(str).str.strip()
    df.loc[df["NAME"].eq(""), "NAME"] = "UNNAMED"

    df["year"] = df["ISO_TIME"].dt.year
    df["month"] = df["ISO_TIME"].dt.month
    df["year_month"] = df["ISO_TIME"].dt.to_period("M").astype(str)
    df["date"] = df["ISO_TIME"].dt.floor("D").dt.tz_localize(None)

    df = add_best_wind_column(df)

    return df.reset_index(drop=True)


def tracks_to_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["LON"], df["LAT"]),
        crs="EPSG:4326",
    )


# ============================================================
# TC-COUNTRY MATCHING
# ============================================================

def build_country_track_hits(
    tracks_gdf: gpd.GeoDataFrame,
    country_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Track-point audit file for one country.

    Keeps IBTrACS track points that intersect the country polygon or fall
    within HIT_BUFFER_KM of it.
    """
    tracks_m = tracks_gdf.to_crs(epsg=3857)
    country_m = country_gdf.to_crs(epsg=3857).copy()

    country_buffered = country_m.copy()
    if HIT_BUFFER_KM > 0:
        country_buffered["geometry"] = country_buffered.geometry.buffer(HIT_BUFFER_KM * 1000)

    hits = gpd.sjoin(
        tracks_m,
        country_buffered[["country", "iso_a3", "geometry"]],
        how="inner",
        predicate="intersects",
    ).drop(columns=["index_right"])

    if hits.empty:
        return pd.DataFrame(
            columns=[
                "SID",
                "NAME",
                "SEASON",
                "ISO_TIME",
                "year",
                "month",
                "year_month",
                "country",
                "iso_a3",
                "best_wind_kt",
                "best_wind_mps",
                "LAT",
                "LON",
            ]
        )

    keep_cols = [
        "SID",
        "NAME",
        "SEASON",
        "ISO_TIME",
        "year",
        "month",
        "year_month",
        "country",
        "iso_a3",
        "best_wind_kt",
        "best_wind_mps",
        "LAT",
        "LON",
    ]

    hits = (
        hits[keep_cols]
        .sort_values(["country", "ISO_TIME", "SID"])
        .reset_index(drop=True)
    )

    return hits


# ============================================================
# STORM-MONTH AND COUNTRY-MONTH INDEX
# ============================================================

def collapse_to_storm_country_month(hits: pd.DataFrame) -> pd.DataFrame:
    """
    One row per storm-country-month.
    """
    if hits.empty:
        return pd.DataFrame(
            columns=[
                "SID",
                "NAME",
                "SEASON",
                "country",
                "iso_a3",
                "year",
                "month",
                "year_month",
                "first_hit_time",
                "last_hit_time",
                "n_track_points",
                "max_wind_kt",
                "max_wind_mps",
            ]
        )

    return (
        hits.groupby(
            ["SID", "NAME", "SEASON", "country", "iso_a3", "year", "month", "year_month"],
            as_index=False,
        )
        .agg(
            first_hit_time=("ISO_TIME", "min"),
            last_hit_time=("ISO_TIME", "max"),
            n_track_points=("ISO_TIME", "count"),
            max_wind_kt=("best_wind_kt", "max"),
            max_wind_mps=("best_wind_mps", "max"),
        )
        .sort_values(["country", "year", "month", "NAME"])
        .reset_index(drop=True)
    )


def collapse_to_country_month_index(storm_month: pd.DataFrame) -> pd.DataFrame:
    """
    One row per country-month with TC/wind exposure variables.
    """
    if storm_month.empty:
        return pd.DataFrame(
            columns=[
                "country",
                "iso_a3",
                "year",
                "month",
                "year_month",
                "n_storms",
                "storm_names",
                "storm_ids",
                "n_track_points_near_country",
                "monthly_max_wind_kt",
                "monthly_max_wind_mps",
            ]
        )

    return (
        storm_month.groupby(
            ["country", "iso_a3", "year", "month", "year_month"],
            as_index=False,
        )
        .agg(
            n_storms=("SID", "nunique"),
            storm_names=("NAME", lambda x: " | ".join(sorted(pd.unique(x)))),
            storm_ids=("SID", lambda x: " | ".join(sorted(pd.unique(x)))),
            n_track_points_near_country=("n_track_points", "sum"),
            monthly_max_wind_kt=("max_wind_kt", "max"),
            monthly_max_wind_mps=("max_wind_mps", "max"),
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
        on=["country", "iso_a3", "year", "month", "year_month"],
        how="left",
    )

    out["n_storms"] = out["n_storms"].fillna(0).astype(int)
    out["storm_names"] = out["storm_names"].fillna("")
    out["storm_ids"] = out["storm_ids"].fillna("")
    out["n_track_points_near_country"] = (
        out["n_track_points_near_country"].fillna(0).astype(int)
    )

    out["monthly_max_wind_kt"] = out["monthly_max_wind_kt"].fillna(0.0)
    out["monthly_max_wind_mps"] = out["monthly_max_wind_mps"].fillna(0.0)

    ordered_cols = [
        "iso3",
        "country",
        "year",
        "month",
        "year_month",
        "date",
        "n_storms",
        "storm_names",
        "storm_ids",
        "n_track_points_near_country",
        "monthly_max_wind_kt",
        "monthly_max_wind_mps",
    ]

    return out[ordered_cols].sort_values(["country", "year", "month"]).reset_index(drop=True)


# ============================================================
# COUNTRY PIPELINE
# ============================================================

def build_country_tc_index(
    tracks_gdf: gpd.GeoDataFrame,
    country_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    country_name = country_gdf["country"].iloc[0]
    iso3 = country_gdf["iso_a3"].iloc[0]

    print(f"\nProcessing {country_name} ({iso3})...")

    hits = build_country_track_hits(tracks_gdf, country_gdf)
    print(f"  Track hits: {len(hits):,}")

    storm_month = collapse_to_storm_country_month(hits)
    print(f"  Storm-country-month rows: {len(storm_month):,}")

    monthly = collapse_to_country_month_index(storm_month)
    balanced = make_balanced_country_month_panel(monthly, country_gdf)

    hits_out = track_hits_file_path(iso3)
    storm_month_out = storm_month_file_path(iso3)
    monthly_out = monthly_index_file_path(iso3)

    hits.to_csv(hits_out, index=False)
    storm_month.to_csv(storm_month_out, index=False)
    balanced.to_csv(monthly_out, index=False)

    print(f"  Saved track-hit audit file: {hits_out}")
    print(f"  Saved storm-month audit file: {storm_month_out}")
    print(f"  Saved monthly TC/wind index: {monthly_out}")

    return balanced


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading Natural Earth countries...")
    countries_gdf = load_countries()
    print(f"Loaded {len(countries_gdf):,} Natural Earth geometries")

    print("Loading IBTrACS...")
    tracks = load_ibtracs()
    print(f"Loaded {len(tracks):,} track rows")

    print("Converting tracks to GeoDataFrame...")
    tracks_gdf = tracks_to_gdf(tracks)

    all_panels = []
    failed = []

    for iso3, country_name in COUNTRIES.items():
        try:
            country_gdf = get_country_polygon(
                countries_gdf=countries_gdf,
                iso3=iso3,
                country_name=country_name,
            )

            panel = build_country_tc_index(
                tracks_gdf=tracks_gdf,
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

        print(f"\nSaved combined monthly TC/wind index: {all_out}")
        print("\nSample:")
        print(all_panel.head(20).to_string(index=False))

    if failed:
        failed_df = pd.DataFrame(failed)
        failed_path = OUT_DIR / "failed_tc_index_builds.csv"
        failed_df.to_csv(failed_path, index=False)

        print(f"\nSaved failure log: {failed_path}")


if __name__ == "__main__":
    main()