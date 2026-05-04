from pathlib import Path
import requests
import pandas as pd


# ============================================================
# Config
# ============================================================

OUTPUT_DIR = Path("data/raw/worldbank")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "government_expenditure_wdi.csv"

INDICATOR = "NE.CON.GOVT.CD"  # General government final consumption expenditure, current US$

COUNTRIES = {
    "DMA": "Dominica",
    "HTI": "Haiti",
    "BHS": "Bahamas, The",
    "BRB": "Barbados",
    "BLZ": "Belize",
    "GRD": "Grenada",
    "JAM": "Jamaica",
    "LCA": "St. Lucia",
    "VCT": "St. Vincent and the Grenadines",
    "TTO": "Trinidad and Tobago",
    "ATG": "Antigua and Barbuda",
    "KNA": "St. Kitts and Nevis",
}


# ============================================================
# Download
# ============================================================

def fetch_world_bank_indicator(country_codes, indicator):
    rows = []

    for iso3, country_name in country_codes.items():
        url = (
            f"https://api.worldbank.org/v2/country/{iso3}/indicator/{indicator}"
            f"?format=json&per_page=20000"
        )

        r = requests.get(url, timeout=60)
        r.raise_for_status()

        payload = r.json()

        if len(payload) < 2 or payload[1] is None:
            print(f"No data returned for {iso3}")
            continue

        for obs in payload[1]:
            rows.append({
                "iso3": iso3,
                "Country": country_name,
                "year": int(obs["date"]),
                "gov_expenditure_usd": obs["value"],
                "indicator": indicator,
                "indicator_name": obs["indicator"]["value"],
            })

    df = pd.DataFrame(rows)

    df["gov_expenditure_usd"] = pd.to_numeric(
        df["gov_expenditure_usd"],
        errors="coerce",
    )

    df = df[df["gov_expenditure_usd"].notna()].copy()

    df["gov_expenditure_musd"] = df["gov_expenditure_usd"] / 1_000_000
    df["monthly_gov_expenditure_musd"] = df["gov_expenditure_musd"] / 12

    df = df.sort_values(["Country", "year"])

    return df


def main():
    df = fetch_world_bank_indicator(COUNTRIES, INDICATOR)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Countries: {df['Country'].nunique()}")
    print(df.tail())


if __name__ == "__main__":
    main()