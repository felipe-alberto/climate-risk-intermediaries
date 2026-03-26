# Under Development by Felipe Verastegui at Columbia IEOR
# Builds three separate World Bank datasets:
#   1) government current consumption
#   2) goods and services expense
#   3) government fixed investment

from pathlib import Path
import requests
import pandas as pd

# -----------------------------
# Config
# -----------------------------
BASE_URL = "https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=20000"
OUTDIR = Path("data/raw/worldbank")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Helper: pull one WB indicator
# -----------------------------
def get_wb_indicator(indicator: str, value_name: str) -> pd.DataFrame:
    url = BASE_URL.format(indicator=indicator)
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    payload = r.json()
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise ValueError(f"No usable data returned for indicator {indicator}")

    data = payload[1]
    df = pd.DataFrame(data)

    # Country field often arrives as a dict: {"id": "CL", "value": "Chile"}
    if "country" in df.columns:
        df["country"] = df["country"].apply(
            lambda x: x["value"] if isinstance(x, dict) and "value" in x else x
        )

    keep_cols = ["country", "countryiso3code", "date", "value"]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Indicator {indicator} missing expected columns: {missing}")

    df = df[keep_cols].copy()
    df.columns = ["country", "country_code", "year", value_name]

    # Basic cleaning
    df = df.dropna(subset=["country_code", "year", value_name])
    df = df[df["country_code"] != ""]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # Drop aggregate regions if you only want sovereign-country style units
    # Keep this commented out if you want WB aggregates too.
    # df = df[df["country_code"].str.len() == 3]

    return df


# -----------------------------
# Pull shared exchange-rate series
# PA.NUS.FCRF = official exchange rate (LCU per US$, period average)
# -----------------------------
fx = get_wb_indicator("PA.NUS.FCRF", "official_fx_lcu_per_usd")
fx = fx[["country_code", "year", "official_fx_lcu_per_usd"]].copy()

# -----------------------------
# 1) Government current consumption
# NE.CON.GOVT.CD = general government final consumption expenditure (current US$)
# -----------------------------
gov_consumption = get_wb_indicator("NE.CON.GOVT.CD", "gov_consumption_usd")
gov_consumption = gov_consumption.sort_values(["country", "year"]).reset_index(drop=True)

print("\n[1] Government current consumption")
print("Rows:", len(gov_consumption))
print("Countries:", gov_consumption["country_code"].nunique())
print(gov_consumption.head(10).to_string(index=False))

gov_consumption.to_csv(
    OUTDIR / "worldbank_gov_current_consumption_usd.csv",
    index=False
)

# -----------------------------
# 2) Goods and services expense
# GC.XPN.GSRV.CN = goods and services expense (current LCU)
# Convert to USD using official exchange rate:
#   USD = LCU / (LCU per US$)
# -----------------------------
goods_services = get_wb_indicator("GC.XPN.GSRV.CN", "goods_services_expense_lcu")
goods_services = goods_services.merge(
    fx,
    on=["country_code", "year"],
    how="left"
)

goods_services["goods_services_expense_usd"] = (
    goods_services["goods_services_expense_lcu"] /
    goods_services["official_fx_lcu_per_usd"]
)

goods_services = goods_services.sort_values(["country", "year"]).reset_index(drop=True)

print("\n[2] Goods and services expense")
print("Rows:", len(goods_services))
print("Countries:", goods_services["country_code"].nunique())
print("Missing FX:", goods_services["official_fx_lcu_per_usd"].isna().sum())
print("Missing USD:", goods_services["goods_services_expense_usd"].isna().sum())
print(goods_services.head(10).to_string(index=False))

goods_services.to_csv(
    OUTDIR / "worldbank_goods_services_expense_lcu_usd.csv",
    index=False
)

# -----------------------------
# 3) General government gross fixed investment
# NE.GDI.FGOV.CN = general government gross domestic fixed investment (current LCU)
# Convert to USD using official exchange rate
# -----------------------------
gov_fixed_investment = get_wb_indicator("NE.GDI.FGOV.CN", "gov_fixed_investment_lcu")
gov_fixed_investment = gov_fixed_investment.merge(
    fx,
    on=["country_code", "year"],
    how="left"
)

gov_fixed_investment["gov_fixed_investment_usd"] = (
    gov_fixed_investment["gov_fixed_investment_lcu"] /
    gov_fixed_investment["official_fx_lcu_per_usd"]
)

gov_fixed_investment = gov_fixed_investment.sort_values(["country", "year"]).reset_index(drop=True)

print("\n[3] Government fixed investment")
print("Rows:", len(gov_fixed_investment))
print("Countries:", gov_fixed_investment["country_code"].nunique())
print("Missing FX:", gov_fixed_investment["official_fx_lcu_per_usd"].isna().sum())
print("Missing USD:", gov_fixed_investment["gov_fixed_investment_usd"].isna().sum())
print(gov_fixed_investment.head(10).to_string(index=False))

gov_fixed_investment.to_csv(
    OUTDIR / "worldbank_gov_fixed_investment_lcu_usd.csv",
    index=False
)


# -----------------------------
# 4) General government expenditure
# NE.CON.GOVT.ZS = general government final consumption expenditure (% of GDP)
# NY.GDP.MKTP.CD = GDP (current US$)
# Compute government expenditure in USD as:
#   gov_exp_usd = (gov_exp_pct_gdp / 100) * gdp_usd
# -----------------------------

gov = get_wb_indicator("NE.CON.GOVT.ZS", "gov_exp_pct_gdp")
gdp = get_wb_indicator("NY.GDP.MKTP.CD", "gdp_usd")
gov = gov[["country", "country_code", "year", "gov_exp_pct_gdp"]].copy()
gdp = gdp[["country_code", "year", "gdp_usd"]].copy()

gov_usd = gov.merge(
    gdp,
    on=["country_code", "year"],
    how="outer"
)
gov_usd["gov_exp_usd"] = gov_usd["gov_exp_pct_gdp"] / 100 * gov_usd["gdp_usd"]


print("Rows:", len(gov_usd))
print("Unique countries:", gov_usd["country_code"].nunique())
print("Missing gov_exp_pct_gdp:", gov_usd["gov_exp_pct_gdp"].isna().sum())
print("Missing gdp_usd:", gov_usd["gdp_usd"].isna().sum())
print("Missing gov_exp_usd:", gov_usd["gov_exp_usd"].isna().sum())
print(gov_usd.head(10).to_string(index=False))

gov_usd = gov_usd.sort_values(["country", "year"])
gov_usd.to_csv(
        OUTDIR / "worldbank_gov_expenditure_usd.csv",
    index=False
    )


print("\nSaved files:")
print("-", OUTDIR / "worldbank_gov_current_consumption_usd.csv")
print("-", OUTDIR / "worldbank_goods_services_expense_lcu_usd.csv")
print("-", OUTDIR / "worldbank_gov_fixed_investment_lcu_usd.csv")
print("-", OUTDIR / "worldbank_gov_expenditure_usd.csv")