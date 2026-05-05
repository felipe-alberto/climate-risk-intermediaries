import pandas as pd

PAYOUT_FILE = "data/raw/climate-risk-pools/payouts.csv"
PANEL_FILE = "data/processed/trigger-proxies/panels/trigger_panel_tc_2007_2024.csv"

COUNTRY_NAME_FIX = {
    "Bahamas, The": "Bahamas",
    "St. Lucia": "Saint Lucia",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
}


def map_policy_to_hazard(policy):
    if pd.isna(policy):
        return "unknown"

    p = str(policy).upper()

    if "XSR" in p:
        return "rain"
    if "TC" in p:
        return "tc"
    if "EQ" in p:
        return "earthquake"

    return "other"


payouts = pd.read_csv(PAYOUT_FILE)

payouts["country"] = payouts["Country"].astype(str).str.strip()
payouts["country"] = payouts["country"].replace(COUNTRY_NAME_FIX)

payouts["year"] = pd.to_numeric(payouts["Year"], errors="coerce")
payouts["month"] = pd.to_numeric(payouts["Month"], errors="coerce")
payouts["day"] = pd.to_numeric(payouts["Day"], errors="coerce").fillna(1)

payouts = payouts.dropna(subset=["year", "month"]).copy()

payouts["plot_date"] = pd.to_datetime(
    dict(year=payouts["year"], month=payouts["month"], day=payouts["day"]),
    errors="coerce",
).dt.to_period("M").dt.to_timestamp()

payouts["hazard"] = payouts["Policy"].apply(map_policy_to_hazard)

raw_tc = payouts[
    (payouts["hazard"] == "tc")
    & (payouts["plot_date"] >= "2007-01-01")
    & (payouts["plot_date"] <= "2024-12-31")
].copy()

raw_tc_months = (
    raw_tc.groupby(["country", "plot_date"], as_index=False)
    .size()
    .rename(columns={"size": "raw_payout_rows"})
)

panel = pd.read_csv(PANEL_FILE)
panel["plot_date"] = pd.to_datetime(panel["plot_date"])

panel_tc_months = (
    panel[panel["has_payout"] == 1]
    .groupby(["country", "plot_date"], as_index=False)
    .size()
    .rename(columns={"size": "panel_rows"})
)

compare = raw_tc_months.merge(
    panel_tc_months,
    on=["country", "plot_date"],
    how="left",
    indicator=True,
)

missing = compare[compare["_merge"] == "left_only"].copy()

print("\n=== RAW TC PAYOUT MONTHS, 2007–2024 ===")
print(len(raw_tc_months))
print(raw_tc_months.sort_values(["plot_date", "country"]).to_string(index=False))

print("\n=== PANEL MATCHED TC PAYOUT MONTHS ===")
print(len(panel_tc_months))
print(panel_tc_months.sort_values(["plot_date", "country"]).to_string(index=False))

print("\n=== MISSING FROM TC PANEL ===")
if missing.empty:
    print("None.")
else:
    print(
        missing[["country", "plot_date", "raw_payout_rows"]]
        .sort_values(["plot_date", "country"])
        .to_string(index=False)
    )

print("\n=== TC PANEL COUNTRIES ===")
print(sorted(panel["country"].unique()))