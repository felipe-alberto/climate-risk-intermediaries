from pathlib import Path
import time
import requests


COUNTRIES = {
    "AIA": "Anguilla",
    "ATG": "Antigua and Barbuda",
    "BHS": "Bahamas",
    "BRB": "Barbados",
    "BLZ": "Belize",
    "BMU": "Bermuda",
    "VGB": "British Virgin Islands",
    "CYM": "Cayman Islands",
    "DMA": "Dominica",
    "DOM": "Dominican Republic",
    "GRD": "Grenada",
    "HTI": "Haiti",
    "JAM": "Jamaica",
    "MSR": "Montserrat",
    "KNA": "Saint Kitts and Nevis",
    "LCA": "Saint Lucia",
    "VCT": "Saint Vincent and the Grenadines",
    "SXM": "Sint Maarten",
    "TTO": "Trinidad and Tobago",
    "TCA": "Turks and Caicos Islands",
}

START_YEAR = 2007
END_YEAR = 2024

OUT_DIR = Path("data/raw/worldpop")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path("data/raw/worldpop/_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

RELEASE = "R2025A"
VERSION = "v1"

BASE_URLS = [
    "https://data.worldpop.org/GIS/Population/Global_2015_2030",
    "https://worldpop-public-data.soton.ac.uk/GIS/Population/Global_2015_2030",
]


def build_filename(iso3: str, year: int) -> str:
    return f"{iso3.lower()}_pop_{year}_CN_1km_{RELEASE}_UA_{VERSION}.tif"


def build_candidate_urls(iso3: str, year: int) -> list[str]:
    filename = build_filename(iso3, year)
    rel_path = f"{RELEASE}/{year}/{iso3}/{VERSION}/1km_ua/constrained/{filename}"
    return [f"{base}/{rel_path}" for base in BASE_URLS]


def find_working_url(iso3: str, year: int) -> str:
    for url in build_candidate_urls(iso3, year):
        try:
            r = requests.head(url, allow_redirects=True, timeout=30)
            if r.status_code == 200:
                return url
        except requests.RequestException:
            continue

    raise FileNotFoundError(f"No working WorldPop URL found for {iso3} in {year}")


def download_file(
    url: str,
    out_path: Path,
    max_retries: int = 3,
    timeout: int = 120,
    chunk_size: int = 1024 * 1024,
) -> Path:
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    if tmp_path.exists():
        tmp_path.unlink()

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[download] attempt {attempt}/{max_retries}: {url}")

            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()

                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

            if tmp_path.stat().st_size == 0:
                raise RuntimeError("Downloaded empty file")

            tmp_path.replace(out_path)

            size_kb = out_path.stat().st_size / 1024
            print(f"[saved] {out_path} ({size_kb:.1f} KB)")
            return out_path

        except Exception as e:
            print(f"[error] {e}")

            if tmp_path.exists():
                tmp_path.unlink()

            if attempt < max_retries:
                time.sleep(5)
            else:
                raise


def download_worldpop_year(
    iso3: str,
    country_name: str,
    year: int,
    overwrite: bool = False,
) -> Path:
    country_dir = OUT_DIR / iso3
    country_dir.mkdir(parents=True, exist_ok=True)

    out_path = country_dir / build_filename(iso3, year)

    if out_path.exists() and not overwrite:
        size_kb = out_path.stat().st_size / 1024
        print(f"[skip] {out_path} ({size_kb:.1f} KB)")
        return out_path

    url = find_working_url(iso3, year)
    return download_file(url, out_path)


def main():
    print("Downloading WorldPop 1km annual population rasters")
    print(f"Countries: {len(COUNTRIES)}")
    print(f"Years: {START_YEAR}-{END_YEAR}")
    print(f"Output directory: {OUT_DIR}\n")

    failures = []

    for iso3, country_name in COUNTRIES.items():
        print("\n==============================")
        print(f"{country_name} ({iso3})")
        print("==============================")

        for year in range(START_YEAR, END_YEAR + 1):
            print(f"\n=== {year} ===")

            try:
                download_worldpop_year(
                    iso3=iso3,
                    country_name=country_name,
                    year=year,
                )

            except Exception as e:
                print(f"[failed] {country_name} ({iso3}), {year}: {e}")

                failures.append(
                    {
                        "iso3": iso3,
                        "country": country_name,
                        "year": year,
                        "error": str(e),
                    }
                )

    if failures:
        import pandas as pd

        failures_df = pd.DataFrame(failures)
        failure_path = LOG_DIR / f"failed_worldpop_downloads_{START_YEAR}_{END_YEAR}.csv"
        failures_df.to_csv(failure_path, index=False)
        print(f"\nSaved failure log: {failure_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()