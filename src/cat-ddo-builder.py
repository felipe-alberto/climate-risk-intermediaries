import requests
import pandas as pd
from pathlib import Path
import json

BASE_URL = "https://search.worldbank.org/api/v2/projects"
OUTDIR = Path("data/raw/worldbank-cat-ddo")
OUTDIR.mkdir(parents=True, exist_ok=True)


def fetch_projects(qterm, rows=500, extra_params=None):
    """
    Query the World Bank Projects API.
    Returns the raw JSON payload.
    """
    params = {
        "format": "json",
        "qterm": qterm,
        "rows": rows,
        "os": 0,          # offset / start
        "apilang": "en",
    }
    if extra_params:
        params.update(extra_params)

    r = requests.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def extract_projects_dict(payload):
    """
    The Projects API usually stores project records in a top-level 'projects' dict,
    keyed by project ID. This helper is slightly defensive.
    """
    if isinstance(payload, dict):
        if "projects" in payload and isinstance(payload["projects"], dict):
            return payload["projects"]
        # fallback: sometimes the payload itself may already be project-like
        if all(isinstance(v, dict) for v in payload.values()):
            return payload
    raise ValueError("Could not find a project dictionary in the API response.")


def first_present(d, candidates):
    """
    Return the first non-empty value among candidate keys.
    """
    for k in candidates:
        if k in d and d[k] not in (None, "", "null"):
            return d[k]
    return None


def cat_ddo_projects():
    """
    Pull likely Cat DDO projects using a few text queries and deduplicate by project_id.
    """
    queries = [
        '"Cat DDO"',
        '"Cat-DDO"',
        '"catastrophe deferred drawdown option"',
        '"DPL Cat DDO"',
    ]

    records = []
    raw_payloads = {}

    for q in queries:
        payload = fetch_projects(qterm=q, rows=500)
        raw_payloads[q] = payload
        projects = extract_projects_dict(payload)

        for pid, p in projects.items():
            rec = {
                "project_id": pid,
                "project_name": first_present(p, ["project_name", "projname", "display_title"]),
                "country": first_present(p, ["countryshortname", "countryname"]),
                "status": first_present(p, ["projectstatusdisplay", "status"]),
                "approval_date": first_present(p, ["boardapprovaldate", "approvalfy", "approvaldate"]),
                "closing_date": first_present(p, ["closingdate"]),
                "lending_instrument": first_present(p, ["lendinginstr", "lendinginstrdisplay"]),
                # amount candidates: inspect raw keys if this comes back missing
                "approved_amount": first_present(
                    p,
                    ["totalamt", "commamt", "commitmentamount", "loanamt", "ibrdcommamt", "idacommamt"]
                ),
                "currency": first_present(p, ["currcode", "currency"]),
                "url": first_present(p, ["url", "projecturl"]),
                "raw_project_name": json.dumps(p.get("project_name")) if "project_name" in p else None,
            }
            records.append(rec)

    df = pd.DataFrame(records).drop_duplicates(subset=["project_id"]).copy()

    # conservative text filter to keep likely Cat DDO operations
    mask = (
        df["project_name"].fillna("").str.contains("cat ddo|cat-ddo|catastrophe deferred drawdown", case=False, regex=True)
        | df["lending_instrument"].fillna("").str.contains("cat ddo|ddo", case=False, regex=True)
    )
    df = df[mask].copy()

    # Numeric clean-up for amount when possible
    if "approved_amount" in df.columns:
        df["approved_amount"] = pd.to_numeric(df["approved_amount"], errors="coerce")

    return df, raw_payloads


if __name__ == "__main__":
    df, raw_payloads = cat_ddo_projects()

    # Save cleaned table
    out_csv = OUTDIR / "worldbank_cat_ddo_projects_first_pass.csv"
    df.sort_values(["country", "project_name"]).to_csv(out_csv, index=False)

    # Save raw payloads for schema inspection
    out_json = OUTDIR / "worldbank_cat_ddo_projects_raw_payloads.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(raw_payloads, f, ensure_ascii=False, indent=2)

    print(f"Found {len(df)} likely Cat DDO projects")
    print(df[["project_id", "country", "project_name", "approved_amount", "currency", "approval_date"]]
          .sort_values(["country", "project_name"])
          .to_string(index=False))

    print("\nSaved:")
    print(out_csv)
    print(out_json)