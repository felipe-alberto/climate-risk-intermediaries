from pathlib import Path
import requests

COUNTRIES = {
    "TTO": "Trinidad and Tobago",
    "JAM": "Jamaica",
    "BRB": "Barbados",
    "BLZ": "Belize",
    "GRD": "Grenada",
    "LCA": "Saint Lucia",
}

START_YEAR = 2017
END_YEAR = 2024

OUT_DIR = Path("data/raw/worldpop")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
            pass

    raise FileNotFoundError(f"No working WorldPop URL found for {iso3} in {year}")


def download_file(url: str, out_path: Path) -> None:
    print(f"Downloading: {url}")

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    size_kb = out_path.stat().st_size / 1024
    print(f"Saved: {out_path} ({size_kb:.1f} KB)")


def download_worldpop_year(iso3: str, country_name: str, year: int) -> Path:
    country_dir = OUT_DIR / iso3
    country_dir.mkdir(parents=True, exist_ok=True)

    out_path = country_dir / build_filename(iso3, year)

    if out_path.exists():
        size_kb = out_path.stat().st_size / 1024
        print(f"Using existing file: {out_path} ({size_kb:.1f} KB)")
        return out_path

    url = find_working_url(iso3, year)
    download_file(url, out_path)
    return out_path


def main():
    print("Downloading WorldPop 1km annual population rasters")
    print(f"Countries: {', '.join(COUNTRIES.values())}")
    print(f"Years: {START_YEAR}-{END_YEAR}")
    print(f"Output directory: {OUT_DIR}\n")

    for iso3, country_name in COUNTRIES.items():
        print(f"\n==============================")
        print(f"{country_name} ({iso3})")
        print(f"==============================")

        for year in range(START_YEAR, END_YEAR + 1):
            print(f"\n=== {year} ===")
            try:
                download_worldpop_year(iso3, country_name, year)
            except FileNotFoundError as e:
                print(f"Skipping: {e}")
            except requests.RequestException as e:
                print(f"Download failed for {country_name} ({iso3}) in {year}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()