"""
Download ERA5 Z500 daily snapshots for a single year from WeatherBench2.

Called by the Snakemake wildcard rule download_era5_z500_year with the year
as a positional argument:

    uv run python pipeline/16_download_era5_z500_daily.py <year>

Output is written to data/downloads/era5/z500_years/<year>.zarr.
All year zarrs are later concatenated by pipeline/17_concat_era5_z500.py.

Source
------
WeatherBench2 ERA5 6-hourly 1.5° (240 × 121 global grid)
gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr

Coverage : 1959-01-02 – 2021-12-31.

Download volume
---------------
WeatherBench global field: 121 × 240 × 4 B ≈ 0.12 MB per step.
~44 MB downloaded per year → ~5 MB stored.

Grid
----
  latitude  : 30.0 – 90.0°N   41 pts  (1.5°)
  longitude : −79.5 – 40.5°E  81 pts  (1.5°)

Time
----
12 UTC daily  =  13:00 CET (UTC+1, no DST).
"""

import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

from wr.paths import ProjPaths

if len(sys.argv) != 2:
    print("Usage: python 16_download_era5_z500_daily.py <year>")
    sys.exit(1)

year     = int(sys.argv[1])
paths    = ProjPaths()
OUT_ZARR = paths.era5_z500_year_zarr(year)

WB_URL = (
    "weatherbench2/datasets/era5/"
    "1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
)
G        = 9.80665
UTC_HOUR = 12

WB_LATS   = np.arange(30.0, 91.5, 1.5)
WB_LONS_W = np.arange(280.5, 360.0, 1.5)
WB_LONS_E = np.arange(0.0, 42.0, 1.5)

if OUT_ZARR.exists():
    print(f"Already cached: {OUT_ZARR}")
    sys.exit(0)

def ts(t0, label):
    elapsed = time.perf_counter() - t0
    print(f"  [{elapsed:5.1f} s] {label}", flush=True)
    return time.perf_counter()

print(f"Downloading {year} from WeatherBench2 ERA5 ...")
t = time.perf_counter()

fs    = gcsfs.GCSFileSystem(token="anon")
store = fs.get_mapper(WB_URL)
ds    = xr.open_zarr(store, consolidated=True)
t = ts(t, "open zarr + read metadata")

times = pd.date_range(f"{year}-01-01 {UTC_HOUR:02d}:00",
                      f"{year}-12-31 {UTC_HOUR:02d}:00", freq="24h")
z = ds["geopotential"].sel(level=500, time=times)
z_w = z.sel(latitude=WB_LATS, longitude=WB_LONS_W, method="nearest")
z_e = z.sel(latitude=WB_LATS, longitude=WB_LONS_E, method="nearest")
z_w = z_w.assign_coords(longitude=z_w.longitude - 360.0)
z_ea = xr.concat([z_w, z_e], dim="longitude").sortby("longitude")
t = ts(t, "build lazy selection")

z500 = (z_ea / G).rename("z500")
z500.attrs = {
    "long_name": "Geopotential height at 500 hPa",
    "units": "gpm",
    "source": f"gs://{WB_URL}",
}
z500 = z500.load()
t = ts(t, "load() — actual GCS transfer")

OUT_ZARR.parent.mkdir(parents=True, exist_ok=True)
z500.to_zarr(str(OUT_ZARR), zarr_format=2)
t = ts(t, "write zarr")

print(f"Saved {OUT_ZARR.name}  {dict(z500.sizes)}  "
      f"Z500: {float(z500.min()):.0f}–{float(z500.max()):.0f} gpm")
