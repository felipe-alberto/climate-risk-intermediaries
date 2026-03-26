import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# ============================================================
# 1. FILE PATHS: CHANGE THESE
# ============================================================

PAYOUTS_FILE = "data/raw/payouts.csv"
GOV_FILE = "data/processed/worldbank_gov_expenditure_usd.csv"
GOV_FILE = "data/processed/worldbank_goods_services_expense_lcu_usd.csv"

# ============================================================
# 2. LOAD DATA
# ============================================================

payouts = pd.read_csv(PAYOUTS_FILE)
gov = pd.read_csv(GOV_FILE)

print("\n=== RAW PAYOUT COLUMNS ===")
print(list(payouts.columns))

print("\n=== RAW GOV COLUMNS ===")
print(list(gov.columns))

# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

payouts.columns = (
    payouts.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

gov.columns = (
    gov.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

print("\n=== CLEANED PAYOUT COLUMNS ===")
print(list(payouts.columns))

print("\n=== CLEANED GOV COLUMNS ===")
print(list(gov.columns))

# ============================================================
# 4. IDENTIFY KEY COLUMNS
# ============================================================

# Expected payout columns from your screenshot:
# country, year, amount_(usd)

if "country" not in payouts.columns:
    raise ValueError("Could not find 'country' column in payouts file.")

if "year" not in payouts.columns:
    raise ValueError("Could not find 'year' column in payouts file.")

# Try to detect amount column robustly
possible_amount_cols = [
    "amount_(usd)",
    "amount_usd",
    "amount",
    "usd_amount",
]

amount_col = None
for col in possible_amount_cols:
    if col in payouts.columns:
        amount_col = col
        break

if amount_col is None:
    raise ValueError(
        f"Could not find payout amount column. Available payout columns: {list(payouts.columns)}"
    )

print(f"\nUsing payout amount column: {amount_col}")

# Expected gov columns from your screenshot:
# country, country_code, year, gov_exp_usd
if "country" not in gov.columns:
    raise ValueError("Could not find 'country' column in gov file.")

if "year" not in gov.columns:
    raise ValueError("Could not find 'year' column in gov file.")

possible_gov_cols = [
    "gov_exp_usd",
    "government_expenditure_usd",
    "gov_expenditure_usd",
    "goods_services_expense_usd"
]

gov_exp_col = None
for col in possible_gov_cols:
    if col in gov.columns:
        gov_exp_col = col
        break

if gov_exp_col is None:
    raise ValueError(
        f"Could not find government expenditure column. Available gov columns: {list(gov.columns)}"
    )

print(f"Using government expenditure column: {gov_exp_col}")

# ============================================================
# 5. CLEAN PAYOUT DATA
# ============================================================

# Keep only necessary columns
payouts = payouts[["country", "year", amount_col]].copy()

# Clean country names
payouts["country"] = payouts["country"].astype(str).str.strip()

# Clean year
payouts["year"] = pd.to_numeric(payouts["year"], errors="coerce")

# Clean amount strings like "$800,000.00"
payouts["amount_usd"] = (
    payouts[amount_col]
    .astype(str)
    .str.replace(r"[\$,]", "", regex=True)
    .str.strip()
)

payouts["amount_usd"] = pd.to_numeric(payouts["amount_usd"], errors="coerce")

# Drop clearly bad rows
payouts = payouts.dropna(subset=["country", "year", "amount_usd"])

# Convert year to int
payouts["year"] = payouts["year"].astype(int)

print("\n=== PAYOUTS AFTER CLEANING ===")
print(payouts.head())

# ============================================================
# 6. AGGREGATE PAYOUTS TO COUNTRY-YEAR
# ============================================================

payouts_agg = (
    payouts.groupby(["country", "year"], as_index=False)["amount_usd"]
    .sum()
    .rename(columns={"amount_usd": "total_payout"})
)

print("\n=== AGGREGATED PAYOUTS ===")
print(payouts_agg.head(20))

# ============================================================
# 7. CLEAN GOV DATA
# ============================================================

# Keep relevant columns only
keep_cols = ["country", "year", gov_exp_col]
if "country_code" in gov.columns:
    keep_cols.append("country_code")

gov = gov[keep_cols].copy()

gov["country"] = gov["country"].astype(str).str.strip()
gov["year"] = pd.to_numeric(gov["year"], errors="coerce")
gov[gov_exp_col] = pd.to_numeric(gov[gov_exp_col], errors="coerce")

gov = gov.dropna(subset=["country", "year", gov_exp_col])
gov["year"] = gov["year"].astype(int)

# Standardize gov expenditure column name
gov = gov.rename(columns={gov_exp_col: "gov_exp_usd"})

print("\n=== GOV DATA AFTER CLEANING ===")
print(gov.head())

# ============================================================
# 8. COUNTRY NAME HARMONIZATION
# ============================================================

name_map = {
    "Bahamas, The": "The Bahamas",
    "The Bahamas": "The Bahamas",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Cote dIvoire": "Cote d'Ivoire",
    "Cote d'Ivoire": "Cote d'Ivoire",
    "Lao People's Democratic Republic": "Lao PDR",
    "Lao PDR": "Lao PDR",
    "St. Lucia": "St. Lucia",
    "Saint Lucia": "St. Lucia",
    "St. Kitts and Nevis": "St. Kitts and Nevis",
    "Saint Kitts and Nevis": "St. Kitts and Nevis",
    "St. Vincent and the Grenadines": "St. Vincent and the Grenadines",
    "Saint Vincent and the Grenadines": "St. Vincent and the Grenadines",
    "Turks and Caicos": "Turks and Caicos Islands",
    "Turks and Caicos Islands": "Turks and Caicos Islands",
}

payouts_agg["country"] = payouts_agg["country"].replace(name_map)
gov["country"] = gov["country"].replace(name_map)

# ============================================================
# 9. OPTIONAL: RESTRICT GOV TO COUNTRIES IN PAYOUT DATA
# ============================================================

countries_with_payouts = sorted(payouts_agg["country"].unique())
gov_sub = gov[gov["country"].isin(countries_with_payouts)].copy()

print("\n=== COUNTRIES IN PAYOUT DATA ===")
print(countries_with_payouts)

print("\nGov rows before restriction:", len(gov))
print("Gov rows after restriction:", len(gov_sub))

# ============================================================
# 10. MERGE
# ============================================================

df = gov_sub.merge(
    payouts_agg,
    on=["country", "year"],
    how="left"
)

df["total_payout"] = df["total_payout"].fillna(0)

# ============================================================
# 11. MERGE CHECKS
# ============================================================

print("\n=== MERGE CHECKS ===")
print("Original aggregated payout rows:", len(payouts_agg))
print("Merged rows with payout > 0:", (df["total_payout"] > 0).sum())

matched_countries = set(df["country"].unique())
missing_countries = set(payouts_agg["country"].unique()) - matched_countries

print("\nCountries in payout data not found in gov data:")
print(missing_countries)

print("\nMissing gov_exp_usd values after merge:")
print(df["gov_exp_usd"].isna().sum())

print("\nSample merged data:")
print(df.sort_values(["country", "year"]).head(30))

# Make patsy/statsmodels happy
df = df.copy()

df["country"] = df["country"].astype(str)
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["total_payout"] = pd.to_numeric(df["total_payout"], errors="coerce")

# Drop bad rows
df = df.dropna(subset=[
    "country", "year", 
]).copy()

# Cast to standard dtypes
df["year"] = df["year"].astype(int)

print("\n=== DTYPES BEFORE REGRESSION ===")
print(df[["country", "year"]].dtypes)

# ============================================================
# 12. CREATE REGRESSION VARIABLES
# ============================================================

# Keep only positive gov expenditure rows
df = df[df["gov_exp_usd"] > 0].copy()

df["payout_dummy"] = (df["total_payout"] > 0).astype(int)
df["log_gov_exp"] = np.log(df["gov_exp_usd"])
df["log_total_payout_plus1"] = np.log(df["total_payout"] + 1)

# Lagged payout dummy
df = df.sort_values(["country", "year"])
df["payout_dummy_lag1"] = df.groupby("country")["payout_dummy"].shift(1)

print("\n=== FINAL REGRESSION DATA ===")
print(df.head(20))

# ============================================================
# 13. SIMPLE SANITY CHECKS
# ============================================================

print("\n=== MEAN GOV EXPENDITURE BY PAYOUT STATUS ===")
print(df.groupby("payout_dummy")["gov_exp_usd"].mean())

print("\n=== COUNT BY PAYOUT STATUS ===")
print(df["payout_dummy"].value_counts(dropna=False))

df["payout_dummy"] = pd.to_numeric(df["payout_dummy"], errors="coerce")
df["payout_dummy"] = df["payout_dummy"].astype(int)

print("\n=== DTYPES BEFORE REGRESSION ===")
print(df[["country", "year", "payout_dummy", "log_gov_exp"]].dtypes)

# ============================================================
# 14. REGRESSION 1: SAME-YEAR PAYOUT DUMMY
# ============================================================

# ============================================================
# 14. REGRESSION 1: SAME-YEAR PAYOUT DUMMY
# ============================================================

import statsmodels.api as sm

reg_df = df[["country", "year", "payout_dummy", "log_gov_exp"]].copy()

reg_df["country"] = reg_df["country"].astype("object")
reg_df["year"] = reg_df["year"].astype(int)
reg_df["payout_dummy"] = reg_df["payout_dummy"].astype(int)
reg_df["log_gov_exp"] = reg_df["log_gov_exp"].astype(float)

X = pd.get_dummies(
    reg_df[["payout_dummy", "country", "year"]],
    columns=["country", "year"],
    drop_first=True
)

X = X.astype(float)
X = sm.add_constant(X)

y = reg_df["log_gov_exp"].astype(float)

model1 = sm.OLS(y, X).fit(
    cov_type="cluster",
    cov_kwds={"groups": reg_df["country"]}
)

print("\n=== REGRESSION 1: log_gov_exp ~ payout_dummy + country FE + year FE ===")
print(model1.summary())

# ============================================================
# 15. REGRESSION 2: LAGGED PAYOUT DUMMY
# ============================================================

df_lag = df.dropna(subset=["payout_dummy_lag1"]).copy()

if len(df_lag) > 0:
    model2 = smf.ols(
        "log_gov_exp ~ payout_dummy_lag1 + C(country) + C(year)",
        data=df_lag
    ).fit(cov_type="cluster", cov_kwds={"groups": df_lag["country"]})

    print("\n=== REGRESSION 2: log_gov_exp ~ payout_dummy_lag1 + FE ===")
    print(model2.summary())
else:
    print("\nNo data available for lagged regression.")

# ============================================================
# 16. SAVE OUTPUT
# ============================================================

df.to_csv("clean_panel_for_regression.csv", index=False)
payouts_agg.to_csv("payouts_country_year.csv", index=False)

print("\nSaved:")
print("- clean_panel_for_regression.csv")
print("- payouts_country_year.csv")