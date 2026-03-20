# Under Development by Felipe Verastegui at Columbia IEOR
# For more information see sites.google.com/view/felipealberto. 

import requests
import pandas as pd

# -----------------------------
# Helper
# -----------------------------
def get_wb_indicator(indicator, value_name):
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=20000"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    data = r.json()[1]
    df = pd.DataFrame(data)

    df["country"] = df["country"].apply(lambda x: x["value"] if isinstance(x, dict) else x)

    df = df[["country", "countryiso3code", "date", "value"]].copy()
    df.columns = ["country", "country_code", "year", value_name]

    df = df.dropna(subset=["country_code", "year", value_name])
    df = df[df["country_code"] != ""]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    return df

# -----------------------------
# Pull WB series
# -----------------------------
gov = get_wb_indicator("NE.CON.GOVT.ZS", "gov_exp_pct_gdp")
gdp = get_wb_indicator("NY.GDP.MKTP.CD", "gdp_usd")

# Keep one country name per code-year from gov
gov = gov[["country", "country_code", "year", "gov_exp_pct_gdp"]].copy()
gdp = gdp[["country_code", "year", "gdp_usd"]].copy()

# -----------------------------
# Merge and compute USD
# -----------------------------
gov_usd = gov.merge(
    gdp,
    on=["country_code", "year"],
    how="outer"
)

gov_usd["gov_exp_usd"] = gov_usd["gov_exp_pct_gdp"] / 100 * gov_usd["gdp_usd"]

# -----------------------------
# Simple sanity checks
# -----------------------------
print("Rows:", len(gov_usd))
print("Unique countries:", gov_usd["country_code"].nunique())
print("Missing gov_exp_pct_gdp:", gov_usd["gov_exp_pct_gdp"].isna().sum())
print("Missing gdp_usd:", gov_usd["gdp_usd"].isna().sum())
print("Missing gov_exp_usd:", gov_usd["gov_exp_usd"].isna().sum())

print("\nSample:")
print(gov_usd.sort_values(["country", "year"]).head(20).to_string(index=False))

# -----------------------------
# Save
# -----------------------------
gov_usd = gov_usd.sort_values(["country", "year"])
gov_usd.to_csv("data/processed/worldbank_gov_expenditure_usd.csv", index=False)

print("\nSaved: data/processed/worldbank_gov_expenditure_usd.csv")