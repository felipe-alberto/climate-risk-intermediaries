"""
Plot threshold-fit time series for trigger proxies.

Updated for multiple candidate index columns per hazard.

Inputs:
    data/processed/trigger-proxies/panels/
        trigger_panel_rain_2007_2024.csv
        trigger_panel_tc_2017_2024.csv
        trigger_panel_earthquake_2007_2024.csv

    data/processed/trigger-proxies/thresholds/
        trigger_thresholds_<hazard>_<index_name>_<start>_<end>.csv

Outputs:
    outputs/figures/trigger-proxies/threshold_fit/
        <hazard>/<index_name>/
            threshold_fit_<hazard>_<index_name>_<ISO3>_<start>_<end>.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

OUT_DIR = Path("outputs/figures/trigger-proxies/threshold_fit")
OUT_DIR.mkdir(parents=True, exist_ok=True)


HAZARD_CONFIGS = {
    "rain": {
        "start_year": 2007,
        "end_year": 2024,
        "panel_file": Path(
            "data/processed/trigger-proxies/panels/"
            "trigger_panel_rain_2007_2024.csv"
        ),
        "threshold_files": [
            Path(
                "data/processed/trigger-proxies/thresholds/"
                "trigger_thresholds_rain_monthly_max_5d_mm_pop_2007_2024.csv"
            ),
            Path(
                "data/processed/trigger-proxies/thresholds/"
                "trigger_thresholds_rain_monthly_max_3d_mm_pop_2007_2024.csv"
            ),
            Path(
                "data/processed/trigger-proxies/thresholds/"
                "trigger_thresholds_rain_monthly_total_mm_pop_2007_2024.csv"
            ),
        ],
        "y_label": "Population-weighted rainfall index (mm)",
    },

    "tc": {
        "start_year": 2007,
        "end_year": 2024,
        "panel_file": Path(
            "data/processed/trigger-proxies/panels/"
            "trigger_panel_tc_2007_2024.csv"
        ),
        "threshold_files": [
            Path(
                "data/processed/trigger-proxies/thresholds/"
                "trigger_thresholds_tc_monthly_max_wind_kt_2007_2024.csv"
            ),
        ],
        "y_label": "Monthly max wind speed near country (kt)",
    },

    "earthquake": {
        "start_year": 2007,
        "end_year": 2024,
        "panel_file": Path(
            "data/processed/trigger-proxies/panels/"
            "trigger_panel_earthquake_2007_2024.csv"
        ),
        "threshold_files": [
            Path(
                "data/processed/trigger-proxies/thresholds/"
                "trigger_thresholds_earthquake_monthly_max_shake_proxy_2007_2024.csv"
            ),
        ],
        "y_label": "Monthly max earthquake shake proxy",
    },
}


# ============================================================
# HELPERS
# ============================================================

def safe_name(x: str) -> str:
    return str(x).replace("/", "_").replace(" ", "_")


# ============================================================
# LOADERS
# ============================================================

def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Panel file not found: {path}")

    df = pd.read_csv(path)

    required = [
        "hazard",
        "iso3",
        "country",
        "plot_date",
        "policy_year",
        "has_payout",
        "payout_amount_usd",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Panel file missing columns: {missing}")

    df["plot_date"] = pd.to_datetime(df["plot_date"])
    df["has_payout"] = (
        pd.to_numeric(df["has_payout"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    df["payout_amount_usd"] = (
        pd.to_numeric(df["payout_amount_usd"], errors="coerce")
        .fillna(0.0)
    )

    return df.sort_values(["country", "plot_date"]).reset_index(drop=True)


def load_thresholds(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Threshold file not found: {path}")

    df = pd.read_csv(path)

    required = [
        "hazard",
        "index_name",
        "iso3",
        "country",
        "policy_year",
        "optimal_threshold",
        "n_payout_months",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Threshold file missing columns: {missing}")

    df["optimal_threshold"] = pd.to_numeric(
        df["optimal_threshold"],
        errors="coerce",
    )

    return df


# ============================================================
# POLICY YEAR HELPERS
# ============================================================

def policy_year_start_date(policy_year: str) -> pd.Timestamp:
    """
    Convert '2024/25' to 2024-06-01.
    """
    start_year = int(str(policy_year).split("/")[0])
    return pd.Timestamp(year=start_year, month=6, day=1)


def policy_year_end_date(policy_year: str) -> pd.Timestamp:
    """
    Convert '2024/25' to 2025-05-31.
    """
    start_year = int(str(policy_year).split("/")[0])
    return pd.Timestamp(year=start_year + 1, month=5, day=31)


def add_policy_year_shading(ax, panel_country: pd.DataFrame) -> None:
    policy_years = (
        panel_country[["policy_year"]]
        .drop_duplicates()
        .sort_values("policy_year")
        ["policy_year"]
        .tolist()
    )

    for i, py in enumerate(policy_years):
        if i % 2 == 0:
            start = max(policy_year_start_date(py), panel_country["plot_date"].min())
            end = min(policy_year_end_date(py), panel_country["plot_date"].max())
            ax.axvspan(start, end, alpha=0.06)


# ============================================================
# PLOTTING
# ============================================================

def plot_threshold_fit_country_hazard_index(
    panel_country: pd.DataFrame,
    thresholds_country: pd.DataFrame,
    hazard: str,
    index_name: str,
    config: dict,
) -> Path:
    panel_country = panel_country.copy()
    thresholds_country = thresholds_country.copy()

    iso3 = panel_country["iso3"].iloc[0]
    country = panel_country["country"].iloc[0]

    if index_name not in panel_country.columns:
        raise ValueError(
            f"Index column '{index_name}' not found in panel for {country} ({iso3})."
        )

    panel_country["hazard_index"] = pd.to_numeric(
        panel_country[index_name],
        errors="coerce",
    )

    panel_country = panel_country.dropna(subset=["hazard_index"]).copy()

    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.plot(
        panel_country["plot_date"],
        panel_country["hazard_index"],
        linewidth=1.8,
        label=index_name,
    )

    payout_months = panel_country[panel_country["has_payout"] == 1].copy()

    if not payout_months.empty:
        ax.scatter(
            payout_months["plot_date"],
            payout_months["hazard_index"],
            s=70,
            marker="o",
            label="Payout month",
            zorder=4,
        )

        for _, row in payout_months.iterrows():
            amount_m = row["payout_amount_usd"] / 1_000_000
            ax.annotate(
                f"${amount_m:.1f}m",
                xy=(row["plot_date"], row["hazard_index"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    add_policy_year_shading(ax, panel_country)

    for _, row in thresholds_country.iterrows():
        threshold = row["optimal_threshold"]

        if pd.isna(threshold):
            continue

        py = row["policy_year"]
        start = max(policy_year_start_date(py), panel_country["plot_date"].min())
        end = min(policy_year_end_date(py), panel_country["plot_date"].max())

        ax.hlines(
            y=threshold,
            xmin=start,
            xmax=end,
            linewidth=2.0,
            linestyles="--",
        )

    title = (
        f"{country} ({iso3}) — {hazard.upper()} threshold fit\n"
        f"Index: {index_name}"
    )

    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(config["y_label"])
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()

    index_dir = OUT_DIR / hazard / safe_name(index_name)
    index_dir.mkdir(parents=True, exist_ok=True)

    out_path = index_dir / (
        f"threshold_fit_{hazard}_{safe_name(index_name)}_"
        f"{iso3}_{config['start_year']}_{config['end_year']}.png"
    )

    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return out_path


def plot_all_for_hazard_index(
    panel: pd.DataFrame,
    thresholds: pd.DataFrame,
    hazard: str,
    index_name: str,
    config: dict,
) -> list[Path]:
    outputs = []

    countries = (
        panel[["iso3", "country"]]
        .drop_duplicates()
        .sort_values(["country", "iso3"])
    )

    for _, row in countries.iterrows():
        iso3 = row["iso3"]
        country = row["country"]

        panel_country = panel[
            (panel["iso3"] == iso3)
            & (panel["country"] == country)
        ].copy()

        thresholds_country = thresholds[
            (thresholds["iso3"] == iso3)
            & (thresholds["country"] == country)
        ].copy()

        if panel_country.empty:
            continue

        if thresholds_country.empty:
            continue

        out_path = plot_threshold_fit_country_hazard_index(
            panel_country=panel_country,
            thresholds_country=thresholds_country,
            hazard=hazard,
            index_name=index_name,
            config=config,
        )

        outputs.append(out_path)

    return outputs


def plot_all_for_hazard(hazard: str, config: dict) -> list[Path]:
    panel = load_panel(config["panel_file"])
    outputs = []

    for threshold_file in config["threshold_files"]:
        thresholds = load_thresholds(threshold_file)

        if thresholds.empty:
            print(f"  Skipping empty threshold file: {threshold_file}")
            continue

        index_names = thresholds["index_name"].dropna().unique()

        if len(index_names) != 1:
            raise ValueError(
                f"Expected one index_name in {threshold_file}, "
                f"found: {index_names}"
            )

        index_name = index_names[0]

        print(f"  Plotting index: {index_name}")

        outputs_index = plot_all_for_hazard_index(
            panel=panel,
            thresholds=thresholds,
            hazard=hazard,
            index_name=index_name,
            config=config,
        )

        outputs.extend(outputs_index)

    return outputs


# ============================================================
# MAIN
# ============================================================

def main():
    all_outputs = []

    for hazard, config in HAZARD_CONFIGS.items():
        print(f"\nPlotting threshold-fit figures for hazard={hazard}")

        outputs = plot_all_for_hazard(hazard, config)

        all_outputs.extend(outputs)
        print(f"  Saved {len(outputs)} figures for hazard={hazard}")

    print(f"\nSaved {len(all_outputs)} total figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()