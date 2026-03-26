from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

COUNTRY_NAME = "Trinidad and Tobago"
START_YEAR = 2017
END_YEAR = 2024

DATA_DIR = Path("data/interim")
FILE_NAME = f"chirps_monthly_metrics_{COUNTRY_NAME.replace(' ', '_')}_{START_YEAR}_{END_YEAR}.csv"
FILE_PATH = DATA_DIR / FILE_NAME


def main():
    if not FILE_PATH.exists():
        raise FileNotFoundError(f"Could not find file: {FILE_PATH}")

    monthly = pd.read_csv(FILE_PATH)

    if "year_month" not in monthly.columns:
        raise ValueError("Expected column 'year_month' not found in input file")

    monthly["date"] = pd.to_datetime(monthly["year_month"])
    monthly = monthly.sort_values("date").reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    plt.plot(monthly["date"], monthly["monthly_total_mm"], label="Monthly total", marker="o")
    plt.plot(monthly["date"], monthly["max_1d_mm"], label="Max 1-day", marker="o")
    plt.plot(monthly["date"], monthly["max_3d_mm"], label="Max 3-day", marker="o")
    plt.plot(monthly["date"], monthly["max_5d_mm"], label="Max 5-day", marker="o")

    plt.title(f"Rainfall metrics over time: {COUNTRY_NAME} ({START_YEAR}-{END_YEAR})")
    plt.xlabel("Date")
    plt.ylabel("Rainfall (mm)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()