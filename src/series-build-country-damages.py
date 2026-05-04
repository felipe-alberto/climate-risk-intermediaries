from pathlib import Path
import pandas as pd


INPUT_PATH = "data/raw/emdat/emdat.xlsx"
OUTPUT_PATH = "data/interim/country-damages/emdat_caribbean_monthly.csv"


CARIBBEAN_COUNTRIES = [
    "Anguilla",
    "Antigua and Barbuda",
    "Bahamas (the)",
    "Bahamas, The",
    "Barbados",
    "Belize",
    "British Virgin Islands",
    "Cayman Islands",
    "Dominica",
    "Dominican Republic",
    "Grenada",
    "Guyana",
    "Haiti",
    "Jamaica",
    "Montserrat",
    "Saint Kitts and Nevis",
    "St. Kitts and Nevis",
    "Saint Lucia",
    "St. Lucia",
    "Saint Vincent and the Grenadines",
    "St. Vincent and the Grenadines",
    "Suriname",
    "Trinidad and Tobago",
    "Turks and Caicos Islands",
]


RELEVANT_HAZARDS = [
    "Storm",
    "Flood",
    "Earthquake",
]


def find_damage_column(df):
    candidates = [
        "Total Damage ('000 US$)",
        "Total Damages ('000 US$)",
        "Total Damage, Adjusted ('000 US$)",
        "Total Damages, Adjusted ('000 US$)",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    matches = [c for c in df.columns if "damage" in c.lower() and "000" in c.lower()]
    if matches:
        return matches[0]

    raise ValueError(
        "Could not find EM-DAT damage column. Available columns:\n"
        + "\n".join(df.columns)
    )


def build_emdat_caribbean_monthly_panel(input_path, output_path):
    input_path = Path(input_path)

    if input_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path, sheet_name="EM-DAT Data")
    else:
        df = pd.read_csv(input_path)

    df.columns = df.columns.str.strip()

    damage_col = find_damage_column(df)

    df = df[df["Country"].isin(CARIBBEAN_COUNTRIES)].copy()
    df = df[df["Disaster Type"].isin(RELEVANT_HAZARDS)].copy()

    df["damage_usd"] = pd.to_numeric(df[damage_col], errors="coerce") * 1_000
    df = df[df["damage_usd"].notna()].copy()

    df["month"] = df["Start Month"].fillna(7)
    df["day"] = df["Start Day"].fillna(1)

    df["date"] = pd.to_datetime(
        {
            "year": df["Start Year"],
            "month": df["month"],
            "day": df["day"],
        },
        errors="coerce",
    )

    df = df[df["date"].notna()].copy()
    df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df
        .groupby(["Country", "year_month"], as_index=False)
        .agg(
            damage_usd=("damage_usd", "sum"),
            n_disasters=("DisNo.", "count"),
        )
    )

    countries = sorted(df["Country"].dropna().unique())

    full_dates = pd.date_range(
        monthly["year_month"].min(),
        monthly["year_month"].max(),
        freq="MS",
    )

    full_index = pd.MultiIndex.from_product(
        [countries, full_dates],
        names=["Country", "year_month"],
    )

    panel = (
        monthly
        .set_index(["Country", "year_month"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    panel["damage_musd"] = panel["damage_usd"] / 1_000_000

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)

    print("Damage column used:", damage_col)
    print("Saved:", output_path)
    print("Rows:", len(panel))
    print("Countries:", panel["Country"].nunique())

    return panel


panel = build_emdat_caribbean_monthly_panel(
    input_path=INPUT_PATH,
    output_path=OUTPUT_PATH,
)