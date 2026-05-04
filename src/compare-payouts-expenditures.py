from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PAYOUTS_PATH = "data/raw/climate-risk-pools/payouts.csv"
EXPENDITURE_PATH = "data/interim/country-expenditure/country_year_expenditure.csv"

OUTPUT_DIR = Path("outputs/descriptive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def clean_money(s):
    return (
        s.astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .replace("", pd.NA)
        .astype(float)
    )


def standardize_country(df, col="Country"):
    return df.assign(
        **{
            col: df[col].replace({
                "Bahamas, The": "Bahamas (the)",
                "St. Lucia": "Saint Lucia",
                "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
                "St. Kitts and Nevis": "Saint Kitts and Nevis",
            })
        }
    )


# ============================================================
# Load payouts
# ============================================================

payouts = pd.read_csv(PAYOUTS_PATH)
payouts = standardize_country(payouts)

payouts["payout_usd"] = clean_money(payouts["Amount (USD)"])
payouts["payout_musd"] = payouts["payout_usd"] / 1_000_000

payouts["date"] = pd.to_datetime(
    {
        "year": payouts["Year"],
        "month": payouts["Month"].fillna(7),
        "day": payouts["Day"].fillna(1),
    },
    errors="coerce",
)

payouts = payouts[payouts["payout_usd"].notna()].copy()
payouts["year"] = payouts["date"].dt.year


payouts_yearly = (
    payouts
    .groupby(["Country", "year"], as_index=False)
    .agg(
        payout_usd=("payout_usd", "sum"),
        payout_musd=("payout_musd", "sum"),
        n_payouts=("payout_usd", "count"),
    )
)


# ============================================================
# Load expenditure
# ============================================================

exp = pd.read_csv(EXPENDITURE_PATH)
exp = standardize_country(exp)


# ============================================================
# Merge
# ============================================================

panel = exp.merge(
    payouts_yearly,
    on=["Country", "year"],
    how="left",
)

panel[["payout_usd", "payout_musd", "n_payouts"]] = panel[
    ["payout_usd", "payout_musd", "n_payouts"]
].fillna(0)


# ============================================================
# Ratios
# ============================================================

panel["payout_over_annual_exp"] = (
    panel["payout_usd"] / panel["gov_expenditure_usd"]
)

panel["payout_over_monthly_exp"] = (
    panel["payout_usd"] / panel["monthly_gov_expenditure_usd"]
)

panel["payout_over_annual_pct"] = 100 * panel["payout_over_annual_exp"]
panel["payout_over_monthly_pct"] = 100 * panel["payout_over_monthly_exp"]


# ============================================================
# Export panel
# ============================================================

panel = panel.sort_values(["Country", "year"])

panel.to_csv(
    OUTPUT_DIR / "country_year_payouts_vs_expenditure.csv",
    index=False,
)


# ============================================================
# Summary stats
# ============================================================

mask = panel["payout_usd"] > 0

summary = pd.DataFrame({
    "Mean payout / annual (%)": [panel.loc[mask, "payout_over_annual_pct"].mean()],
    "Median payout / annual (%)": [panel.loc[mask, "payout_over_annual_pct"].median()],
    "Mean payout / monthly (%)": [panel.loc[mask, "payout_over_monthly_pct"].mean()],
    "Median payout / monthly (%)": [panel.loc[mask, "payout_over_monthly_pct"].median()],
    "Max payout / annual (%)": [panel["payout_over_annual_pct"].max()],
    "Max payout / monthly (%)": [panel["payout_over_monthly_pct"].max()],
})

summary = summary.round(2)

summary.to_csv(
    OUTPUT_DIR / "summary_payouts_vs_expenditure.csv",
    index=False,
)

summary.to_latex(
    OUTPUT_DIR / "summary_payouts_vs_expenditure.tex",
    index=False,
    float_format="%.2f",
    caption="Parametric insurance payouts relative to government expenditure.",
    label="tab:payouts_vs_expenditure",
)


# ============================================================
# Figure 1: scatter (annual)
# ============================================================

plt.figure(figsize=(6.5, 5.5))

plt.scatter(
    panel["gov_expenditure_musd"],
    panel["payout_musd"],
    alpha=0.75,
)

max_val = max(
    panel["gov_expenditure_musd"].max(),
    panel["payout_musd"].max(),
)

plt.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1)

plt.xlabel("Government expenditure (USD millions)")
plt.ylabel("Payouts (USD millions)")
plt.title("Payouts vs Government Expenditure")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "scatter_payouts_vs_expenditure.png", dpi=300)
plt.savefig(OUTPUT_DIR / "scatter_payouts_vs_expenditure.pdf")

plt.close()


# ============================================================
# Figure 2: distribution of ratios
# ============================================================

plt.figure(figsize=(6, 4))

plt.hist(
    panel.loc[mask, "payout_over_annual_pct"],
    bins=30,
)

plt.xlabel("Payout / annual expenditure (%)")
plt.ylabel("Frequency")
plt.title("Distribution of Payout-to-Expenditure Ratios")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "hist_payout_over_expenditure.png", dpi=300)
plt.close()


print("Saved outputs to:", OUTPUT_DIR)
print(summary)