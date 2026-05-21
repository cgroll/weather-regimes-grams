"""
Concatenate per-year ERA5 Z500 zarrs into a single zarr.

Reads every <year>.zarr from data/downloads/era5/z500_years/, concatenates
along the time dimension in chronological order, and writes the result to
data/downloads/era5/z500_euro_atlantic.zarr.

Run after all desired years have been downloaded by
pipeline/16_download_era5_z500_daily.py.
"""

import xarray as xr

from wr.paths import ProjPaths

paths    = ProjPaths()
OUT_ZARR = paths.era5_z500_daily_zarr

year_zarrs = sorted(paths.era5_z500_years_path.glob("*.zarr"))
if not year_zarrs:
    raise FileNotFoundError(f"No year zarrs found in {paths.era5_z500_years_path}")

years = [int(p.stem) for p in year_zarrs]
print(f"Concatenating {len(years)} years: {years[0]}–{years[-1]}")

datasets = [xr.open_zarr(str(p)) for p in year_zarrs]
combined = xr.concat(datasets, dim="time")

print(f"Combined shape: {dict(combined.sizes)}")
print(f"Writing → {OUT_ZARR} ...")
OUT_ZARR.parent.mkdir(parents=True, exist_ok=True)
combined.to_zarr(str(OUT_ZARR), zarr_format=2)
print("Done.")
