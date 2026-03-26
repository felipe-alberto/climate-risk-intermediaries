from pathlib import Path
import requests

COUNTRY_NAME = "Trinidad and Tobago"
ISO3 = "TTO"
START_YEAR = 2017
END_YEAR = 2024

OUT_DIR = Path("data/raw/worldpop")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Current WorldPop Global2 public release
RELEASE = "R2025A"
VERSION = "v1"

BASE_URLS = [
    "https://data.worldpop.org/GIS/Population/Global_2015_2030",
    "https://worldpop-public-data.soton.ac.uk/GIS/Population/Global_2015_2030",
]


def build_filename(year: int) -> str:
    return f"{ISO3.lower()}_pop_{year}_CN_1km_{RELEASE}_UA_{VERSION}.tif"


def build_candidate_urls(year: int) -> list[str]:
    filename = build_filename(year)
    rel_path = f"{RELEASE}/{year}/{ISO3}/{VERSION}/1km_ua/constrained/{filename}"
    return [f"{base}/{rel_path}" for base in BASE_URLS]


def find_working_url(year: int) -> str:
    for url in build_candidate_urls(year):
        try:
            r = requests.head(url, allow_redirects=True, timeout=30)
            if r.status_code == 200:
                return url
        except requests.RequestException:
            pass
    raise FileNotFoundError(f"No working WorldPop URL found for {ISO3} in {year}")


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


def download_worldpop_year(year: int) -> Path:
    out_path = OUT_DIR / build_filename(year)

    if out_path.exists():
        size_kb = out_path.stat().st_size / 1024
        print(f"Using existing file: {out_path} ({size_kb:.1f} KB)")
        return out_path

    url = find_working_url(year)
    download_file(url, out_path)
    return out_path


def main():
    print(f"Downloading WorldPop 1km annual population rasters for {COUNTRY_NAME} ({ISO3})")
    print(f"Years: {START_YEAR}-{END_YEAR}")
    print(f"Output directory: {OUT_DIR}\n")

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"=== {year} ===")
        download_worldpop_year(year)

    print("\nDone.")


if __name__ == "__main__":
    main()