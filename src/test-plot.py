from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

COUNTRY_NAME = "Trinidad and Tobago"
START_YEAR = 2017
END_YEAR = 2024

INTERIM_DIR = Path("data/interim")
RAW_DIR = Path("data/raw")

RAIN_FILE = INTERIM_DIR / f"chirps_monthly_metrics_{COUNTRY_NAME.replace(' ', '_')}_{START_YEAR}_{END_YEAR}.csv"
PAYOUT_FILE = RAW_DIR / "payouts.csv"


def clean_amount_usd(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA})
        .astype(float)
    )


def load_rainfall() -> pd.DataFrame:
    if not RAIN_FILE.exists():
        raise FileNotFoundError(f"Rainfall file not found: {RAIN_FILE}")

    monthly = pd.read_csv(RAIN_FILE)
    monthly["date"] = pd.to_datetime(monthly["year_month"])
    monthly = monthly.sort_values("date").reset_index(drop=True)
    return monthly


def load_country_payouts() -> pd.DataFrame:
    if not PAYOUT_FILE.exists():
        raise FileNotFoundError(f"Payout file not found: {PAYOUT_FILE}")

    payouts = pd.read_csv(PAYOUT_FILE)

    payouts = payouts[payouts["Country"] == COUNTRY_NAME].copy()

    if payouts.empty:
        print(f"No payouts found for {COUNTRY_NAME}")
        return payouts

    # Clean amount
    payouts["amount_usd"] = clean_amount_usd(payouts["Amount (USD)"])

    # Some rows may have missing day; default to 1 so we can place them on the monthly timeline
    payouts["Day"] = pd.to_numeric(payouts["Day"], errors="coerce").fillna(1).astype(int)
    payouts["Month"] = pd.to_numeric(payouts["Month"], errors="coerce").astype(int)
    payouts["Year"] = pd.to_numeric(payouts["Year"], errors="coerce").astype(int)

    payouts["date"] = pd.to_datetime(
        dict(year=payouts["Year"], month=payouts["Month"], day=payouts["Day"]),
        errors="coerce"
    )

    payouts = payouts.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Keep only the rainfall sample window
    start_date = pd.Timestamp(f"{START_YEAR}-01-01")
    end_date = pd.Timestamp(f"{END_YEAR}-12-31")
    payouts = payouts[(payouts["date"] >= start_date) & (payouts["date"] <= end_date)].copy()

    return payouts


def add_payout_lines(ax, payouts: pd.DataFrame) -> None:
    if payouts.empty:
        return

    for _, row in payouts.iterrows():
        label = f"{row['Pool']} payout" if pd.notna(row.get("Pool")) else "Payout"
        ax.axvline(row["date"], linestyle="--", alpha=0.6)

    # Add only one legend entry
    ax.axvline(payouts.iloc[0]["date"], linestyle="--", alpha=0.6, label="Payout date")


def plot_monthly_total_with_payouts(monthly: pd.DataFrame, payouts: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(monthly["date"], monthly["monthly_total_mm"], marker="o", label="Monthly total rainfall")
    add_payout_lines(ax1, payouts)

    ax1.set_title(f"Monthly rainfall and payouts: {COUNTRY_NAME} ({START_YEAR}-{END_YEAR})")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Rainfall (mm)")
    ax1.tick_params(axis="x", rotation=45)

    # Right axis for payout amounts
    ax2 = ax1.twinx()
    if not payouts.empty:
        ax2.scatter(payouts["date"], payouts["amount_usd"], marker="o", label="Payout amount")
    ax2.set_ylabel("Payout amount (USD)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    plt.show()


def plot_extremes_with_payouts(monthly: pd.DataFrame, payouts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    series_info = [
        ("max_1d_mm", "Max 1-day rainfall"),
        ("max_3d_mm", "Max 3-day rainfall"),
        ("max_5d_mm", "Max 5-day rainfall"),
    ]

    for ax, (col, title) in zip(axes, series_info):
        ax.plot(monthly["date"], monthly[col], marker="o")
        add_payout_lines(ax, payouts)
        ax.set_title(title)
        ax.set_ylabel("Rainfall (mm)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    axes[-1].tick_params(axis="x", rotation=45)

    fig.suptitle(f"Extreme rainfall metrics and payouts: {COUNTRY_NAME} ({START_YEAR}-{END_YEAR})")
    fig.tight_layout()
    plt.show()


def main():
    monthly = load_rainfall()
    payouts = load_country_payouts()

    print(f"Loaded {len(monthly)} monthly rainfall observations")
    print(f"Loaded {len(payouts)} payouts for {COUNTRY_NAME}")

    plot_monthly_total_with_payouts(monthly, payouts)
    plot_extremes_with_payouts(monthly, payouts)


if __name__ == "__main__":
    main()