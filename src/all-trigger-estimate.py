"""
Evaluate threshold fit for disaster-risk-pool payout proxies.

Outputs:
    data/processed/trigger-proxies/
        panels/
            trigger_panel_<hazard>_<start>_<end>.csv

        thresholds/
            trigger_thresholds_<hazard>_<start>_<end>.csv

        trigger_thresholds_all_hazards.csv
        trigger_summary_country_hazard.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

PAYOUT_FILE = Path("data/raw/climate-risk-pools/payouts.csv")

OUT_DIR = Path("data/processed/trigger-proxies")
PANEL_OUT_DIR = OUT_DIR / "panels"
THRESHOLD_OUT_DIR = OUT_DIR / "thresholds"

PANEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
THRESHOLD_OUT_DIR.mkdir(parents=True, exist_ok=True)


HAZARD_CONFIGS = {
    "rain": {
        "start_year": 2017,
        "end_year": 2024,
        "panel_file": Path(
            "data/interim/rain-index/"
            "chirps_monthly_metrics_popweighted_all_countries_2017_2024.csv"
        ),
        "index_column": "monthly_total_mm_pop",
        "payout_keywords": ["rain", "rainfall", "excess rainfall", "xsr"],
    },
    "tc": {
        "start_year": 2017,
        "end_year": 2024,
        "panel_file": Path(
            "data/interim/tc-index/"
            "tc_country_month_panel_all_countries_2017_2024.csv"
        ),
        "index_column": "monthly_max_wind_kt",
        "payout_keywords": ["tropical cyclone", "cyclone", "hurricane", "wind", "tc"],
    },
    "earthquake": {
        "start_year": 2007,
        "end_year": 2024,
        "panel_file": Path(
            "data/interim/earthquake-index/"
            "eq_country_month_panel_all_countries_2007_2024.csv"
        ),
        "index_column": "monthly_max_shake_proxy",
        "payout_keywords": ["earthquake", "eq"],
    },
}


# ============================================================
# HELPERS
# ============================================================

def map_policy_to_hazard(policy: str) -> str:
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

def clean_amount_usd(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA}),
        errors="coerce",
    )

def get_payouts_for_hazard(payouts: pd.DataFrame, hazard: str) -> pd.DataFrame:
    return payouts[payouts["hazard"] == hazard].copy()


def get_policy_year_label(date: pd.Timestamp) -> str:
    """
    CCRIF policy year runs June 1 to May 31.

    Example:
        2024-07 -> 2024/25
        2025-03 -> 2024/25
    """
    if date.month >= 6:
        start_year = date.year
    else:
        start_year = date.year - 1

    end_year_short = str((start_year + 1) % 100).zfill(2)
    return f"{start_year}/{end_year_short}"


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def output_panel_path(hazard: str, config: dict) -> Path:
    return PANEL_OUT_DIR / (
        f"trigger_panel_{hazard}_{config['start_year']}_{config['end_year']}.csv"
    )


def output_threshold_path(hazard: str, config: dict) -> Path:
    return THRESHOLD_OUT_DIR / (
        f"trigger_thresholds_{hazard}_{config['start_year']}_{config['end_year']}.csv"
    )


# ============================================================
# LOADERS
# ============================================================

def load_hazard_panel(hazard: str, config: dict) -> pd.DataFrame:
    path = config["panel_file"]
    index_col = config["index_column"]

    if not path.exists():
        raise FileNotFoundError(f"{hazard} panel file not found: {path}")

    df = pd.read_csv(path)

    required = ["iso3", "country", "year", "month", index_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{hazard} panel missing columns: {missing}")

    if "date" in df.columns:
        df["plot_date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    elif "year_month" in df.columns:
        df["plot_date"] = pd.to_datetime(df["year_month"]).dt.to_period("M").dt.to_timestamp()
    else:
        df["plot_date"] = pd.to_datetime(
            dict(year=df["year"], month=df["month"], day=1)
        )

    start_date = pd.Timestamp(f"{config['start_year']}-01-01")
    end_date = pd.Timestamp(f"{config['end_year']}-12-31")

    df = df[
        (df["plot_date"] >= start_date)
        & (df["plot_date"] <= end_date)
    ].copy()

    df["policy_year"] = df["plot_date"].apply(get_policy_year_label)
    df["hazard"] = hazard
    df["hazard_index"] = pd.to_numeric(df[index_col], errors="coerce")

    keep_cols = [
        "hazard",
        "iso3",
        "country",
        "year",
        "month",
        "plot_date",
        "policy_year",
        "hazard_index",
    ]

    optional_cols = [
        "monthly_total_mm_pop",
        "monthly_max_wind_kt",
        "monthly_max_wind_mps",
        "n_storms",
        "storm_names",
        "storm_ids",
        "monthly_max_shake_proxy",
        "monthly_sum_shake_proxy",
        "monthly_max_mag_nearby",
        "n_eq_nearby",
    ]

    keep_cols += [c for c in optional_cols if c in df.columns and c not in keep_cols]

    return df[keep_cols].sort_values(["country", "plot_date"]).reset_index(drop=True)


def load_payouts() -> pd.DataFrame:
    payouts = pd.read_csv(PAYOUT_FILE)

    payouts["country"] = payouts["Country"].astype(str).str.strip()
    payouts["amount_usd"] = clean_amount_usd(payouts["Amount (USD)"])

    payouts["Year"] = pd.to_numeric(payouts["Year"], errors="coerce")
    payouts["Month"] = pd.to_numeric(payouts["Month"], errors="coerce")
    payouts["Day"] = pd.to_numeric(payouts["Day"], errors="coerce").fillna(1)

    payouts = payouts.dropna(subset=["Year", "Month"]).copy()

    payouts["date"] = pd.to_datetime(
        dict(year=payouts["Year"], month=payouts["Month"], day=payouts["Day"]),
        errors="coerce",
    )

    payouts = payouts.dropna(subset=["date"]).copy()

    payouts["plot_date"] = payouts["date"].dt.to_period("M").dt.to_timestamp()
    payouts["policy_year"] = payouts["plot_date"].apply(get_policy_year_label)

    # Classification based on Policy
    payouts["policy_clean"] = payouts["Policy"].astype(str).str.strip().str.upper()

    # Classification based on insured policy/product, not event type
    payouts["policy_clean"] = payouts["Policy"].astype(str).str.strip().str.upper()
    payouts["hazard"] = payouts["policy_clean"].apply(map_policy_to_hazard)

    # Keep only core hazards used in trigger-proxy evaluation
    payouts = payouts[payouts["hazard"].isin(["rain", "tc", "earthquake"])].copy()

    return payouts.sort_values(["country", "plot_date"]).reset_index(drop=True)



def restrict_payouts_to_hazard_window(
    payouts: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    start_date = pd.Timestamp(f"{config['start_year']}-01-01")
    end_date = pd.Timestamp(f"{config['end_year']}-12-31")

    return payouts[
        (payouts["date"] >= start_date)
        & (payouts["date"] <= end_date)
    ].copy()


# ============================================================
# PANEL CONSTRUCTION
# ============================================================

def build_hazard_payout_panel(
    hazard_panel: pd.DataFrame,
    payouts_hazard: pd.DataFrame,
) -> pd.DataFrame:
    panel = hazard_panel.copy()

    if payouts_hazard.empty:
        panel["has_payout"] = 0
        panel["payout_amount_usd"] = 0.0
        return panel

    payout_months = (
        payouts_hazard.groupby(["country", "plot_date"], as_index=False)
        .agg(
            has_payout=("amount_usd", lambda x: 1),
            payout_amount_usd=("amount_usd", "sum"),
        )
    )

    panel = panel.merge(
        payout_months,
        on=["country", "plot_date"],
        how="left",
    )

    panel["has_payout"] = panel["has_payout"].fillna(0).astype(int)
    panel["payout_amount_usd"] = panel["payout_amount_usd"].fillna(0.0)

    return panel.sort_values(["country", "plot_date"]).reset_index(drop=True)


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

NEAR_BELOW_BANDWIDTH_PCTS = [0.05, 0.10]


def add_threshold_flags(
    g: pd.DataFrame,
    threshold: float,
    bandwidth_pcts: list[float] = NEAR_BELOW_BANDWIDTH_PCTS,
) -> pd.DataFrame:
    """
    Add classification and below-threshold near-control flags.

    Canonical classification:
        TP: index >= threshold and payout
        FP: index >= threshold and no payout
        FN: index <  threshold and payout
        TN: index <  threshold and no payout

    RDD logic:
        - All FP months are useful untreated high-index months:
              index >= threshold, no payout.

        - TN months are useful only if they are close below the threshold:
              index in [(1 - bw) * threshold, threshold), no payout.

    Therefore, the bandwidth is applied only below the threshold.
    """
    g = g.copy()

    g["above_threshold"] = g["hazard_index"] >= threshold
    g["below_threshold"] = g["hazard_index"] < threshold

    g["tp"] = (
        g["above_threshold"]
        & (g["has_payout"] == 1)
    ).astype(int)

    g["fp"] = (
        g["above_threshold"]
        & (g["has_payout"] == 0)
    ).astype(int)

    g["fn"] = (
        g["below_threshold"]
        & (g["has_payout"] == 1)
    ).astype(int)

    g["tn"] = (
        g["below_threshold"]
        & (g["has_payout"] == 0)
    ).astype(int)

    for bw in bandwidth_pcts:
        label = int(round(100 * bw))
        lower = threshold * (1 - bw)

        g[f"near_below_{label}pct"] = (
            (g["hazard_index"] >= lower)
            & (g["hazard_index"] < threshold)
        ).astype(int)

        g[f"near_tn_below_{label}pct"] = (
            (g[f"near_below_{label}pct"] == 1)
            & (g["tn"] == 1)
        ).astype(int)

        g[f"near_fn_below_{label}pct"] = (
            (g[f"near_below_{label}pct"] == 1)
            & (g["fn"] == 1)
        ).astype(int)

    return g


def score_threshold(g: pd.DataFrame, threshold: float) -> dict:
    g_scored = add_threshold_flags(g, threshold)

    tp = int(g_scored["tp"].sum())
    fp = int(g_scored["fp"].sum())
    fn = int(g_scored["fn"].sum())
    tn = int(g_scored["tn"].sum())

    n = len(g_scored)

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    balanced_accuracy = (
        0.5 * (sensitivity + specificity)
        if pd.notna(sensitivity) and pd.notna(specificity)
        else np.nan
    )

    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if pd.notna(precision)
        and pd.notna(sensitivity)
        and (precision + sensitivity) > 0
        else np.nan
    )

    return {
        "threshold": threshold,
        "n_months": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "correct_total": tp + tn,
        "incorrect_total": fp + fn,
        "accuracy": (tp + tn) / n if n > 0 else np.nan,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "net_score": (tp + tn) - (fp + fn),
    }


def summarize_below_threshold_near_controls(
    g: pd.DataFrame,
    threshold: float,
    bandwidth_pcts: list[float] = NEAR_BELOW_BANDWIDTH_PCTS,
) -> dict:
    """
    Summarize below-threshold near-control diagnostics.

    near_tn_below_Xpct:
        close-below-threshold, no payout.
        These are your most useful untreated near-miss months.

    near_fn_below_Xpct:
        close-below-threshold, payout.
        These are below-threshold treated months close to the cutoff.
    """
    g_scored = add_threshold_flags(g, threshold, bandwidth_pcts)

    out = {}

    for bw in bandwidth_pcts:
        label = int(round(100 * bw))

        near_below_n = int(g_scored[f"near_below_{label}pct"].sum())
        near_tn = int(g_scored[f"near_tn_below_{label}pct"].sum())
        near_fn = int(g_scored[f"near_fn_below_{label}pct"].sum())

        near_below_payout_rate = (
            near_fn / near_below_n
            if near_below_n > 0
            else np.nan
        )

        out.update({
            f"near_below_{label}pct_n": near_below_n,
            f"near_tn_below_{label}pct": near_tn,
            f"near_fn_below_{label}pct": near_fn,
            f"near_below_payout_rate_{label}pct": near_below_payout_rate,
        })

    return out


def empty_threshold_result(base: dict) -> dict:
    out = {
        **base,
        "optimal_threshold": np.nan,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "precision": np.nan,
        "sensitivity": np.nan,
        "specificity": np.nan,
        "f1": np.nan,
        "net_score": np.nan,
        "tp": np.nan,
        "fp": np.nan,
        "fn": np.nan,
        "tn": np.nan,
        "above_threshold_with_payout": np.nan,
        "above_threshold_no_payout": np.nan,
        "below_threshold_with_payout": np.nan,
        "below_threshold_no_payout": np.nan,
        "rdd_treated_n": np.nan,
        "rdd_untreated_5pct_n": np.nan,
        "rdd_untreated_10pct_n": np.nan,
    }

    for bw in NEAR_BELOW_BANDWIDTH_PCTS:
        label = int(round(100 * bw))
        out.update({
            f"near_below_{label}pct_n": np.nan,
            f"near_tn_below_{label}pct": np.nan,
            f"near_fn_below_{label}pct": np.nan,
            f"near_below_payout_rate_{label}pct": np.nan,
        })

    return out


def find_optimal_threshold_for_group(g: pd.DataFrame) -> dict:
    g = g.sort_values("plot_date").copy()

    hazard = g["hazard"].iloc[0]
    iso3 = g["iso3"].iloc[0]
    country = g["country"].iloc[0]
    policy_year = g["policy_year"].iloc[0]

    n_payouts = int(g["has_payout"].sum())

    base = {
        "hazard": hazard,
        "iso3": iso3,
        "country": country,
        "policy_year": policy_year,
        "n_months": len(g),
        "n_payout_months": n_payouts,
    }

    if n_payouts == 0:
        return empty_threshold_result(base)

    candidate_thresholds = sorted(g["hazard_index"].dropna().unique())

    if len(candidate_thresholds) == 0:
        return empty_threshold_result(base)

    scores = [score_threshold(g, threshold=t) for t in candidate_thresholds]
    scores_df = pd.DataFrame(scores)

    best = (
        scores_df
        .sort_values(
            by=[
                "balanced_accuracy",
                "accuracy",
                "fn",
                "fp",
                "threshold",
            ],
            ascending=[False, False, True, True, True],
        )
        .iloc[0]
    )

    threshold = float(best["threshold"])
    near_counts = summarize_below_threshold_near_controls(g, threshold)

    tp = int(best["tp"])
    fp = int(best["fp"])
    fn = int(best["fn"])
    tn = int(best["tn"])

    return {
        **base,
        "optimal_threshold": threshold,
        "accuracy": best["accuracy"],
        "balanced_accuracy": best["balanced_accuracy"],
        "precision": best["precision"],
        "sensitivity": best["sensitivity"],
        "specificity": best["specificity"],
        "f1": best["f1"],
        "net_score": best["net_score"],

        # Canonical confusion matrix
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,

        # Readable aliases
        "above_threshold_with_payout": tp,
        "above_threshold_no_payout": fp,
        "below_threshold_with_payout": fn,
        "below_threshold_no_payout": tn,

        # RDD interpretation
        # Treated = all payout months, whether above or below the rule threshold.
        "rdd_treated_n": tp + fn,

        # Untreated = all above-threshold no-payout months
        # plus close-below-threshold no-payout months.
        "rdd_untreated_5pct_n": fp + near_counts.get("near_tn_below_5pct", 0),
        "rdd_untreated_10pct_n": fp + near_counts.get("near_tn_below_10pct", 0),

        # Below-threshold near-control diagnostics
        **near_counts,
    }


def estimate_thresholds(panel: pd.DataFrame) -> pd.DataFrame:
    results = []

    group_cols = ["hazard", "iso3", "country", "policy_year"]

    for _, g in panel.groupby(group_cols, sort=True):
        results.append(find_optimal_threshold_for_group(g))

    return pd.DataFrame(results)

# ============================================================
# SUMMARY TABLES
# ============================================================

def summarize_country_hazard(thresholds: pd.DataFrame) -> pd.DataFrame:
    valid = thresholds[thresholds["n_payout_months"] > 0].copy()

    if valid.empty:
        return pd.DataFrame()

    return (
        valid.groupby(["iso3", "country", "hazard"], as_index=False)
        .agg(
            n_policy_years_with_payouts=("policy_year", "nunique"),
            total_payout_months=("n_payout_months", "sum"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_precision=("precision", "mean"),
            mean_sensitivity=("sensitivity", "mean"),
            mean_specificity=("specificity", "mean"),
            mean_f1=("f1", "mean"),
        )
        .sort_values(["country", "hazard"])
        .reset_index(drop=True)
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Using payout file: {PAYOUT_FILE}")

    payouts = load_payouts()
    print(f"Loaded {len(payouts):,} payout rows")

    all_thresholds = []

    for hazard, config in HAZARD_CONFIGS.items():
        print(f"\n=== Processing hazard: {hazard} ===")
        print(f"Window: {config['start_year']}–{config['end_year']}")
        print(f"Using panel file: {config['panel_file']}")

        hazard_panel = load_hazard_panel(hazard, config)
        payouts_hazard = get_payouts_for_hazard(payouts, hazard)
        payouts_hazard = restrict_payouts_to_hazard_window(payouts_hazard, config)

        print(f"Loaded {len(hazard_panel):,} hazard-panel rows")
        print(f"Matched {len(payouts_hazard):,} payout rows for hazard={hazard}")

        panel = build_hazard_payout_panel(
            hazard_panel=hazard_panel,
            payouts_hazard=payouts_hazard,
        )

        thresholds = estimate_thresholds(panel)

        panel_out = output_panel_path(hazard, config)
        thresholds_out = output_threshold_path(hazard, config)

        panel.to_csv(panel_out, index=False)
        thresholds.to_csv(thresholds_out, index=False)

        print(f"Saved hazard payout panel: {panel_out}")
        print(f"Saved thresholds:          {thresholds_out}")

        all_thresholds.append(thresholds)

    if all_thresholds:
        thresholds_all = pd.concat(all_thresholds, ignore_index=True)

        thresholds_all_out = OUT_DIR / "trigger_thresholds_all_hazards.csv"
        thresholds_all.to_csv(thresholds_all_out, index=False)

        summary = summarize_country_hazard(thresholds_all)

        summary_out = OUT_DIR / "trigger_summary_country_hazard.csv"
        summary.to_csv(summary_out, index=False)

        print(f"\nSaved all-hazard thresholds: {thresholds_all_out}")
        print(f"Saved country-hazard summary: {summary_out}")

        print("\nCountry-hazard summary:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()