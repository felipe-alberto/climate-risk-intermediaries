# Under Development by Felipe Verastegui at Columbia IEOR
# For more information see sites.google.com/view/felipealberto.

import requests
import pandas as pd

# -----------------------------
# 1. Load and clean payouts
# -----------------------------
payouts = pd.read_csv("data/raw/payouts.csv")

payouts = payouts[["ID", "Pool", "Year", "Country", "Amount (USD)"]].copy()

payouts["Year"] = pd.to_numeric(payouts["Year"], errors="coerce")
payouts["Amount (USD)"] = (
    payouts["Amount (USD)"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)
payouts["Amount (USD)"] = pd.to_numeric(payouts["Amount (USD)"], errors="coerce")

print("Rows in payouts:", len(payouts))
print("Missing Year:", payouts["Year"].isna().sum())
print("Missing Amount:", payouts["Amount (USD)"].isna().sum())
print("Duplicate IDs:", payouts["ID"].duplicated().sum())

payouts = payouts.dropna(subset=["Year", "Amount (USD)"]).copy()
payouts["Year"] = payouts["Year"].astype(int)

# -----------------------------
# 2. Helper to pull WB data
# -----------------------------
def get_wb_indicator(indicator, value_name):
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=20000"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    payload = r.json()
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise ValueError(f"Unexpected response for indicator {indicator}")

    df = pd.DataFrame(payload[1])

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
# 3. Build country crosswalk
# -----------------------------
gov_raw = get_wb_indicator("NE.CON.GOVT.ZS", "gov_exp_pct_gdp")
country_crosswalk = gov_raw[["country", "country_code"]].drop_duplicates()

manual_codes = {
    "Anguilla": "AIA",
    "The Bahamas": "BHS",
}

payouts = payouts.merge(
    country_crosswalk,
    left_on="Country",
    right_on="country",
    how="left"
)

payouts["country_code"] = payouts["country_code"].fillna(
    payouts["Country"].map(manual_codes)
)

unmatched = (
    payouts[payouts["country_code"].isna()]["Country"]
    .drop_duplicates()
    .sort_values()
)
print("\nUnmatched countries:")
print(unmatched.to_list())

payouts = payouts[payouts["country_code"].notna()].copy()

# -----------------------------
# 4. Collapse to country-year
# -----------------------------
country_year_payouts = (
    payouts
    .groupby(["country_code", "Country", "Year"], as_index=False)["Amount (USD)"]
    .sum()
    .rename(columns={
        "Country": "country",
        "Year": "year",
        "Amount (USD)": "total_payout_usd"
    })
)

print("\nCountry-year payout rows:", len(country_year_payouts))

# -----------------------------
# 5. Pull full gov expenditure + GDP
# -----------------------------
gov = get_wb_indicator("NE.CON.GOVT.ZS", "gov_exp_pct_gdp")
gdp = get_wb_indicator("NY.GDP.MKTP.CD", "gdp_usd")

gov_full = gov.merge(
    gdp[["country_code", "year", "gdp_usd"]],
    on=["country_code", "year"],
    how="inner"
)

gov_full["gov_exp_usd"] = gov_full["gov_exp_pct_gdp"] / 100 * gov_full["gdp_usd"]

# -----------------------------
# 6. Pull goods/services expense + exchange rate
#    WB indicator:
#    GC.XPN.GSRV.CN = Goods and services expense (current LCU)
#    PA.NUS.FCRF    = Official exchange rate (LCU per US$, period average)
# -----------------------------
goods_services_lcu = get_wb_indicator("GC.XPN.GSRV.CN", "goods_services_lcu")
fx = get_wb_indicator("PA.NUS.FCRF", "lcu_per_usd")

goods_services_full = goods_services_lcu.merge(
    fx[["country_code", "year", "lcu_per_usd"]],
    on=["country_code", "year"],
    how="left"
)

goods_services_full["goods_services_usd"] = (
    goods_services_full["goods_services_lcu"] / goods_services_full["lcu_per_usd"]
)

# -----------------------------
# 7. Merge everything
# -----------------------------
merged = country_year_payouts.merge(
    gov_full[["country_code", "year", "gov_exp_usd", "gov_exp_pct_gdp"]],
    on=["country_code", "year"],
    how="left"
)

merged = merged.merge(
    goods_services_full[
        ["country_code", "year", "goods_services_lcu", "lcu_per_usd", "goods_services_usd"]
    ],
    on=["country_code", "year"],
    how="left"
)

print("Rows after merge:", len(merged))
print("Missing gov_exp_usd:", merged["gov_exp_usd"].isna().sum())
print("Missing goods_services_usd:", merged["goods_services_usd"].isna().sum())

missing_gov = merged[merged["gov_exp_usd"].isna()][["country", "country_code", "year"]]
if len(missing_gov) > 0:
    print("\nMissing full government expenditure rows:")
    print(missing_gov.to_string(index=False))

missing_gs = merged[merged["goods_services_usd"].isna()][["country", "country_code", "year"]]
if len(missing_gs) > 0:
    print("\nMissing goods/services rows:")
    print(missing_gs.to_string(index=False))

# -----------------------------
# 8. Compute payout shares
# -----------------------------
merged["payout_share_of_gov_exp"] = merged["total_payout_usd"] / merged["gov_exp_usd"]
merged["payout_pct_of_gov_exp"] = 100 * merged["payout_share_of_gov_exp"]

merged["payout_share_of_goods_services"] = (
    merged["total_payout_usd"] / merged["goods_services_usd"]
)
merged["payout_pct_of_goods_services"] = 100 * merged["payout_share_of_goods_services"]

# -----------------------------
# 9. Summaries
# -----------------------------
print("\nSummary of payout % of full government expenditure:")
print(merged["payout_pct_of_gov_exp"].describe())

print("\nSummary of payout % of goods/services expense:")
print(merged["payout_pct_of_goods_services"].describe())

print("\nLargest payout shares vs full government expenditure:")
print(
    merged[
        ["country", "year", "total_payout_usd", "gov_exp_usd", "payout_pct_of_gov_exp"]
    ]
    .sort_values("payout_pct_of_gov_exp", ascending=False)
    .head(15)
    .to_string(index=False)
)

print("\nLargest payout shares vs goods/services expense:")
print(
    merged[
        ["country", "year", "total_payout_usd", "goods_services_usd", "payout_pct_of_goods_services"]
    ]
    .sort_values("payout_pct_of_goods_services", ascending=False)
    .head(15)
    .to_string(index=False)
)

# -----------------------------
# 10. Save
# -----------------------------
merged.to_csv("data/interim/payouts_vs_gov_expenditure_and_goods_services.csv", index=False)
print("\nSaved: data/interim/payouts_vs_gov_expenditure_and_goods_services.csv")