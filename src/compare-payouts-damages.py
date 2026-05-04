from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PAYOUTS_PATH = "data/raw/climate-risk-pools/payouts.csv"
DAMAGES_PATH = "data/interim/country-damages/emdat_caribbean_monthly.csv"

OUTPUT_DIR = Path("outputs/descriptive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def clean_money_column(s):
    return (
        s.astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .replace("", pd.NA)
        .astype(float)
    )


def standardize_country_names(df, col="Country"):
    replacements = {
        "Bahamas, The": "Bahamas (the)",
        "St. Lucia": "Saint Lucia",
        "St. Kitts and Nevis": "Saint Kitts and Nevis",
        "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    }

    df[col] = df[col].replace(replacements)
    return df


# ============================================================
# Load payouts
# ============================================================

payouts = pd.read_csv(PAYOUTS_PATH)
payouts = standardize_country_names(payouts)

payouts["payout_usd"] = clean_money_column(payouts["Amount (USD)"])
payouts["payout_musd"] = payouts["payout_usd"] / 1_000_000

payouts["date"] = pd.to_datetime(
    {
        "year": payouts["Year"],
        "month": payouts["Month"].fillna(7),
        "day": payouts["Day"].fillna(1),
    },
    errors="coerce",
)

payouts = payouts[payouts["payout_musd"] > 0].copy()
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
# Load damages
# ============================================================

damages = pd.read_csv(DAMAGES_PATH)
damages = standardize_country_names(damages)

damages["year_month"] = pd.to_datetime(damages["year_month"])
damages["year"] = damages["year_month"].dt.year

damages_yearly = (
    damages
    .groupby(["Country", "year"], as_index=False)
    .agg(
        damage_usd=("damage_usd", "sum"),
        damage_musd=("damage_musd", "sum"),
        n_disasters=("n_disasters", "sum"),
    )
)


# ============================================================
# Merge
# ============================================================

panel = damages_yearly.merge(
    payouts_yearly,
    on=["Country", "year"],
    how="outer",
)

for col in [
    "damage_usd", "damage_musd", "n_disasters",
    "payout_usd", "payout_musd", "n_payouts"
]:
    panel[col] = panel[col].fillna(0)


# ============================================================
# Coverage ratios
# ============================================================

panel["coverage_ratio"] = pd.NA
mask = panel["damage_usd"] > 0

panel.loc[mask, "coverage_ratio"] = (
    panel.loc[mask, "payout_usd"] / panel.loc[mask, "damage_usd"]
)

panel["coverage_pct"] = panel["coverage_ratio"] * 100

panel["payout_minus_damage_musd"] = (
    panel["payout_musd"] - panel["damage_musd"]
)


# ============================================================
# Export country-year panel
# ============================================================

panel = panel.sort_values(["Country", "year"])

panel.to_csv(
    OUTPUT_DIR / "yearly_payouts_vs_emdat_damages.csv",
    index=False,
)


# ============================================================
# Summary table
# ============================================================

summary = pd.DataFrame({
    "Total damage, USD millions": [panel["damage_musd"].sum()],
    "Total payout, USD millions": [panel["payout_musd"].sum()],
    "Aggregate payout / damage (%)": [
        100 * panel["payout_usd"].sum() / panel["damage_usd"].sum()
        if panel["damage_usd"].sum() > 0 else pd.NA
    ],
    "Country-years with damage": [(panel["damage_usd"] > 0).sum()],
    "Country-years with payout": [(panel["payout_usd"] > 0).sum()],
    "Country-years with both": [
        ((panel["damage_usd"] > 0) & (panel["payout_usd"] > 0)).sum()
    ],
})

summary = summary.round(2)

summary.to_csv(
    OUTPUT_DIR / "summary_payouts_vs_emdat_damages.csv",
    index=False,
)

summary.to_latex(
    OUTPUT_DIR / "summary_payouts_vs_emdat_damages.tex",
    index=False,
    float_format="%.2f",
    caption="Parametric insurance payouts relative to EM-DAT disaster damages.",
    label="tab:payouts_vs_damages",
)


# ============================================================
# Country-level totals
# ============================================================

country_totals = (
    panel
    .groupby("Country", as_index=False)
    .agg(
        damage_musd=("damage_musd", "sum"),
        payout_musd=("payout_musd", "sum"),
        n_disasters=("n_disasters", "sum"),
        n_payouts=("n_payouts", "sum"),
    )
)

country_totals["coverage_pct"] = (
    100 * country_totals["payout_musd"] / country_totals["damage_musd"]
)

country_totals.loc[
    country_totals["damage_musd"] == 0, "coverage_pct"
] = pd.NA

country_totals = country_totals.sort_values("damage_musd", ascending=False)

country_totals.to_csv(
    OUTPUT_DIR / "country_totals_payouts_vs_emdat_damages.csv",
    index=False,
)


# ============================================================
# Figure: payout vs damage
# ============================================================

plot_df = panel[
    (panel["damage_musd"] > 0) | (panel["payout_musd"] > 0)
].copy()

plt.figure(figsize=(6.5, 5.5))

plt.scatter(
    plot_df["damage_musd"],
    plot_df["payout_musd"],
    alpha=0.75,
    s=45,
)

max_val = max(
    plot_df["damage_musd"].max(),
    plot_df["payout_musd"].max(),
)

plt.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1)

plt.xlabel("Annual EM-DAT damages (USD millions)")
plt.ylabel("Annual parametric payouts (USD millions)")
plt.title("Parametric Payouts vs. Disaster Damages")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "scatter_payouts_vs_emdat_damages.png", dpi=300)
plt.savefig(OUTPUT_DIR / "scatter_payouts_vs_emdat_damages.pdf")

plt.close()


print("Saved comparison outputs to:", OUTPUT_DIR)
print(summary)