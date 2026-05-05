import pandas as pd

panel = pd.read_csv(
    "data/processed/trigger-proxies/panels/trigger_panel_tc_2007_2024.csv"
)
panel["plot_date"] = pd.to_datetime(panel["plot_date"])

targets = [
    ("Saint Lucia", "2010-10-01"),
    ("Saint Vincent and the Grenadines", "2010-10-01"),
    ("Saint Kitts and Nevis", "2017-09-01"),
    ("Saint Lucia", "2017-09-01"),
    ("Bahamas", "2019-09-01"),
    ("Saint Vincent and the Grenadines", "2024-07-01"),
]

for country, date in targets:
    hit = panel[
        (panel["country"] == country)
        & (panel["plot_date"] == pd.Timestamp(date))
    ]

    print("\n", country, date)
    if hit.empty:
        print("  NO PANEL ROW")
    else:
        print(hit[["country", "plot_date", "has_payout", "payout_amount_usd"]].to_string(index=False))