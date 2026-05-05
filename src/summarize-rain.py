from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = Path(
    "data/interim/rain-index/"
    "chirps_monthly_rain_indices_popweighted_all_countries_2007_2024.csv"
)

OUT_DIR = Path("outputs/descriptive/rain")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_COUNTRIES = [
    "JAM",  # Jamaica
    "HTI",  # Haiti
    "DOM",  # Dominican Republic
    "TTO",  # Trinidad and Tobago
    "BRB",  # Barbados
]

RAIN_INDEX_COLUMNS = [
    "monthly_total_mm_pop",
    "monthly_max_1d_mm_pop",
    "monthly_max_3d_mm_pop",
    "monthly_max_5d_mm_pop",
    "monthly_p95_daily_mm_pop",
    "monthly_p99_daily_mm_pop",
]


# ============================================================
# HELPERS
# ============================================================

def summarize_country(df: pd.DataFrame) -> pd.Series:
    out = {"n_months": len(df)}

    for col in RAIN_INDEX_COLUMNS:
        s = df[col].dropna()

        out[f"{col}_mean"] = s.mean()
        out[f"{col}_std"] = s.std()
        out[f"{col}_p50"] = s.quantile(0.50)
        out[f"{col}_p90"] = s.quantile(0.90)
        out[f"{col}_p95"] = s.quantile(0.95)
        out[f"{col}_p99"] = s.quantile(0.99)
        out[f"{col}_max"] = s.max()

    return pd.Series(out)


def save_selected_country_summary(df: pd.DataFrame) -> None:
    summary = (
        df.groupby(["iso3", "country"])
        .apply(summarize_country)
        .reset_index()
    )

    out_file = OUT_DIR / "rain_summary_selected_countries.csv"
    summary.to_csv(out_file, index=False)

    print("\nSummary table:")
    print(summary.to_string(index=False))
    print(f"\nSaved summary: {out_file}")


def plot_time_series(df: pd.DataFrame, metric: str) -> None:
    plt.figure(figsize=(12, 6))

    for iso3, g in df.groupby("iso3"):
        g = g.sort_values("date")
        country = g["country"].iloc[0]
        plt.plot(g["date"], g[metric], label=f"{country} ({iso3})", linewidth=1.5)

    plt.title(f"Monthly rainfall index: {metric}")
    plt.xlabel("Date")
    plt.ylabel("Rainfall index, mm")
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_file = OUT_DIR / f"time_series_{metric}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved plot: {out_file}")


def plot_country_distribution(df: pd.DataFrame, metric: str) -> None:
    plot_df = df[["iso3", "country", metric]].dropna().copy()

    labels = (
        plot_df[["iso3", "country"]]
        .drop_duplicates()
        .sort_values("iso3")
    )

    data = []
    tick_labels = []

    for _, row in labels.iterrows():
        iso3 = row["iso3"]
        country = row["country"]
        values = plot_df.loc[plot_df["iso3"] == iso3, metric].values

        data.append(values)
        tick_labels.append(f"{iso3}")

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=tick_labels, showfliers=True)

    plt.title(f"Distribution of monthly rainfall index: {metric}")
    plt.xlabel("Country")
    plt.ylabel("Rainfall index, mm")
    plt.tight_layout()

    out_file = OUT_DIR / f"boxplot_{metric}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved plot: {out_file}")


def plot_metric_correlation(df: pd.DataFrame) -> None:
    corr = df[RAIN_INDEX_COLUMNS].corr()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(range(len(RAIN_INDEX_COLUMNS)), RAIN_INDEX_COLUMNS, rotation=90)
    plt.yticks(range(len(RAIN_INDEX_COLUMNS)), RAIN_INDEX_COLUMNS)

    plt.title("Correlation across rainfall index candidates")
    plt.tight_layout()

    out_file = OUT_DIR / "rain_index_correlation_matrix.png"
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved plot: {out_file}")


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = df[df["iso3"].isin(SELECTED_COUNTRIES)].copy()

    if df.empty:
        raise ValueError("No rows found for selected countries.")

    save_selected_country_summary(df)

    # Time-series plots
    plot_time_series(df, "monthly_total_mm_pop")
    plot_time_series(df, "monthly_max_1d_mm_pop")
    plot_time_series(df, "monthly_max_3d_mm_pop")
    plot_time_series(df, "monthly_max_5d_mm_pop")

    # Distribution plots
    plot_country_distribution(df, "monthly_total_mm_pop")
    plot_country_distribution(df, "monthly_max_1d_mm_pop")
    plot_country_distribution(df, "monthly_max_5d_mm_pop")

    # Correlation among candidate indices
    plot_metric_correlation(df)

    print("\nDone.")


if __name__ == "__main__":
    main()