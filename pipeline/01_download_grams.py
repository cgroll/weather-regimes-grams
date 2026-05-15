"""Download Christian Grams weather regimes dataset from Zenodo (record 17080146).

Pure data script — no visualizations. Downloads the full archive, extracts it
into data/downloads/, then removes both zip files. To force a fresh download,
delete data/downloads/wr_data_package_V1.0/ and re-run snakemake.
"""

import urllib.request
import zipfile
from wr.paths import ProjPaths

ZENODO_URL = "https://zenodo.org/api/records/17080146/files-archive"

paths = ProjPaths()
paths.ensure_directories()

zip_path = paths.downloads_path / "grams_weather_regimes.zip"

print(f"Downloading {ZENODO_URL} ...")
urllib.request.urlretrieve(ZENODO_URL, zip_path)
print(f"Saved → {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

print("Extracting outer archive ...")
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(paths.downloads_path)
    names = zf.namelist()
print(f"Extracted {len(names)} files → {paths.downloads_path}")

inner_zip = paths.downloads_path / "wr_data_package_V1.0.zip"
if inner_zip.exists():
    print("Extracting inner data package ...")
    with zipfile.ZipFile(inner_zip, "r") as zf:
        zf.extractall(paths.downloads_path)
        inner_names = zf.namelist()
    print(f"Extracted {len(inner_names)} files → {paths.downloads_path}")
    inner_zip.unlink()
    print(f"Removed {inner_zip.name}")

zip_path.unlink()
print(f"Removed {zip_path.name}")
