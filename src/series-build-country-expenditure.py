from pathlib import Path
import pandas as pd


# ============================================================
# Paths
# ============================================================

INPUT_PATH = "data/raw/worldbank/government_expenditure_wdi.csv"
OUTPUT_PATH = "data/interim/country-expenditure/country_year_expenditure.csv"


# ============================================================
# Country-name standardization
# ============================================================

COUNTRY_REPLACEMENTS = {
    "Bahamas, The": "Bahamas (the)",
    "St. Lucia": "Saint Lucia",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
}


def build_country_year_expenditure_panel(
    input_path: str = INPUT_PATH,
    output_path: str = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Build country-year government expenditure panel from World Bank WDI data.

    Input:
        data/raw/worldbank/government_expenditure_wdi.csv

    Output:
        data/interim/country-expenditure/country_year_expenditure.csv

    Variables:
        Country
        iso3
        year
        gov_expenditure_usd
        gov_expenditure_musd
        monthly_gov_expenditure_usd
        monthly_gov_expenditure_musd
    """

    df = pd.read_csv(input_path)

    # ------------------------------------------------------------
    # Basic cleaning
    # ------------------------------------------------------------

    df["Country"] = df["Country"].replace(COUNTRY_REPLACEMENTS)

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    df["gov_expenditure_usd"] = pd.to_numeric(
        df["gov_expenditure_usd"],
        errors="coerce",
    )

    df = df[
        df["year"].notna()
        & df["gov_expenditure_usd"].notna()
        & (df["gov_expenditure_usd"] > 0)
    ].copy()

    # ------------------------------------------------------------
    # Construct scaled variables
    # ------------------------------------------------------------

    df["gov_expenditure_musd"] = df["gov_expenditure_usd"] / 1_000_000

    df["monthly_gov_expenditure_usd"] = df["gov_expenditure_usd"] / 12
    df["monthly_gov_expenditure_musd"] = df["gov_expenditure_musd"] / 12

    # ------------------------------------------------------------
    # Keep clean panel columns
    # ------------------------------------------------------------

    keep_cols = [
        "Country",
        "iso3",
        "year",
        "gov_expenditure_usd",
        "gov_expenditure_musd",
        "monthly_gov_expenditure_usd",
        "monthly_gov_expenditure_musd",
    ]

    panel = df[keep_cols].copy()

    panel = panel.sort_values(["Country", "year"]).reset_index(drop=True)

    # ------------------------------------------------------------
    # Export
    # ------------------------------------------------------------

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panel.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(panel):,}")
    print(f"Countries: {panel['Country'].nunique()}")
    print(f"Years: {panel['year'].min()}–{panel['year'].max()}")

    return panel


if __name__ == "__main__":
    build_country_year_expenditure_panel()