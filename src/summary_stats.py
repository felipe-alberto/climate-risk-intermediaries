from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

INPUT_PATH = "data/raw/climate-risk-pools/payouts.csv"
OUTPUT_DIR = Path("outputs/descriptive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load and clean
# ============================================================

df = pd.read_csv(INPUT_PATH)

df["amount_usd"] = (
    df["Amount (USD)"]
    .astype(str)
    .str.replace(r"[^\d.]", "", regex=True)
    .replace("", pd.NA)
    .astype(float)
)

df["amount_musd"] = df["amount_usd"] / 1_000_000

df["date"] = pd.to_datetime(
    {
        "year": df["Year"],
        "month": df["Month"].fillna(7),
        "day": df["Day"].fillna(1),
    },
    errors="coerce",
)

df_pos = df[df["amount_musd"] > 0].copy()


# ============================================================
# Summary statistics
# ============================================================

def summary_stats(x: pd.Series) -> pd.Series:
    return pd.Series({
        "N": x.count(),
        "Mean": x.mean(),
        "Std. Dev.": x.std(),
        "Median": x.median(),
        "P25": x.quantile(0.25),
        "P75": x.quantile(0.75),
        "Min": x.min(),
        "Max": x.max(),
        "Total": x.sum(),
    })


def export_table(table: pd.DataFrame, name: str) -> pd.DataFrame:
    table = table.round(2)

    table.to_csv(OUTPUT_DIR / f"{name}.csv")

    table.to_latex(
        OUTPUT_DIR / f"{name}.tex",
        float_format="%.2f",
        caption="Summary statistics of parametric insurance payouts. "
                "Payouts are in USD millions and conditional on payout > 0.",
        label=f"tab:{name}",
    )

    return table


table_overall = summary_stats(df_pos["amount_musd"]).to_frame("All payouts").T
export_table(table_overall, "summary_overall")

table_by_pool = (
    df_pos
    .groupby("Pool")["amount_musd"]
    .apply(summary_stats)
    .unstack()
    .sort_values("Total", ascending=False)
)
export_table(table_by_pool, "summary_by_pool")

table_by_pool_policy = (
    df_pos
    .groupby(["Pool", "Policy"], dropna=False)["amount_musd"]
    .apply(summary_stats)
    .unstack()
    .sort_values("Total", ascending=False)
)
export_table(table_by_pool_policy, "summary_by_pool_policy")


# ============================================================
# Monthly aggregation
# ============================================================

df_pos["year_month"] = df_pos["date"].dt.to_period("M").dt.to_timestamp()

monthly = (
    df_pos
    .groupby(["year_month", "Pool"], as_index=False)["amount_musd"]
    .sum()
)


# ============================================================
# Figure 1: monthly scatter by pool
# ============================================================

plt.figure(figsize=(9, 5))

for pool, g in monthly.groupby("Pool"):
    plt.scatter(
        g["year_month"],
        g["amount_musd"],
        label=pool,
        alpha=0.8,
        s=45,
    )

plt.xlabel("Month")
plt.ylabel("Monthly payout (USD millions)")
plt.title("Monthly Parametric Insurance Payouts by Pool")
plt.legend(title="Pool", frameon=False)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "monthly_payouts_scatter_by_pool.png", dpi=300)
plt.savefig(OUTPUT_DIR / "monthly_payouts_scatter_by_pool.pdf")
plt.close()


# ============================================================
# Figure 2: monthly line plot by pool
# Optional; use if you want a cash-flow time-series view
# ============================================================

plt.figure(figsize=(9, 5))

for pool, g in monthly.groupby("Pool"):
    plt.plot(
        g["year_month"],
        g["amount_musd"],
        marker="o",
        linewidth=1,
        label=pool,
    )

plt.xlabel("Month")
plt.ylabel("Monthly payout (USD millions)")
plt.title("Monthly Parametric Insurance Payouts by Pool")
plt.legend(title="Pool", frameon=False)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "monthly_payouts_line_by_pool.png", dpi=300)
plt.savefig(OUTPUT_DIR / "monthly_payouts_line_by_pool.pdf")
plt.close()


print("Saved descriptive tables and figures to:", OUTPUT_DIR)