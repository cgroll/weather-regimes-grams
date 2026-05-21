"""
Download ERA5 Z500 daily climatology from WeatherBench2 (1.5° global grid).

Computes the day-of-year mean from the WeatherBench2 6-hourly ERA5 reanalysis
for the Grams Euro-Atlantic domain.

Source
------
WeatherBench2 ERA5 6-hourly at 1.5° (240 × 121 global grid)
gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr

Domain
------
Grams WR EOF domain (nearest 1.5° grid points):
  latitude  : 30.0 – 90.0°N   41 points  (1.5° step)
  longitude : −79.5 – 40.5°E  81 points  (1.5° step)

Time
----
12 UTC daily  =  13:00 CET (UTC+1, no DST).
WeatherBench is 6-hourly (00/06/12/18 UTC); 12 UTC is the closest standard
synoptic hour to local noon in Germany.

Reference period : 1979-01-01 – 2019-12-31  (matches Grams dataset)
Climatology      : day-of-year mean (groupby dayofyear, standard for
                   anomaly computation in meteorology)

Output
------
data/downloads/wb/z500_climatology.zarr
  variable  z500        geopotential height at 500 hPa (gpm)
  dims      dayofyear × latitude × longitude
"""

import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

from wr.paths import ProjPaths

paths = ProjPaths()
OUT_ZARR = paths.wb_z500_climatology

# WeatherBench2 ERA5 6-hourly 1.5° (240×121)
# Check https://weatherbench2.readthedocs.io for the latest available zarr path.
WB_URL = (
    "weatherbench2/datasets/era5/"
    "1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
)

G         = 9.80665   # m/s²
UTC_HOUR  = 12        # 12 UTC = 13:00 CET
REF_START = "1979-01-01"
REF_END   = "2019-12-31"

# Euro-Atlantic domain on the WeatherBench 1.5° grid.
# These lon values are in 0–360 convention (WeatherBench native).
# 280.5° = −79.5°E,  40.5° = 40.5°E — nearest 1.5° points to the Grams −80/40 domain.
# All of these coincide exactly with ARCO 0.25° grid points (1.5 / 0.25 = 6).
WB_LATS   = np.arange(30.0, 91.5, 1.5)         # 30 to 90  (41 pts, ascending)
WB_LONS_W = np.arange(280.5, 360.0, 1.5)       # 280.5–358.5  (53 pts, western block)
WB_LONS_E = np.arange(0.0, 42.0, 1.5)          # 0–40.5       (28 pts, eastern block)

# ── Cache check ────────────────────────────────────────────────────────────────
if OUT_ZARR.exists():
    print(f"Using cached climatology: {OUT_ZARR}")
else:
    print(f"Opening WeatherBench2 ERA5 zarr: gs://{WB_URL}")
    fs    = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(WB_URL)
    ds    = xr.open_zarr(store, consolidated=True)

    print(f"  Variables : {list(ds.data_vars)}")
    print(f"  Lat       : {float(ds.latitude[0]):.1f} → {float(ds.latitude[-1]):.1f}  "
          f"({len(ds.latitude)} pts)")
    print(f"  Lon       : {float(ds.longitude[0]):.1f} → {float(ds.longitude[-1]):.1f}  "
          f"({len(ds.longitude)} pts)")
    print(f"  Time      : {str(ds.time.values[0])[:13]} → {str(ds.time.values[-1])[:13]}")

    # ── Select Z500 at 12 UTC for the reference period ────────────────────────
    times_12utc = pd.date_range(
        f"{REF_START} {UTC_HOUR:02d}:00",
        f"{REF_END}   {UTC_HOUR:02d}:00",
        freq="24h",
    )
    print(f"\n  Selecting {len(times_12utc)} daily 12 UTC time steps (1979–2019) ...")

    z = ds["geopotential"].sel(level=500, time=times_12utc)

    # ── Select Euro-Atlantic domain (west and east of 0°, then merge) ─────────
    z_w = z.sel(latitude=WB_LATS, longitude=WB_LONS_W)
    z_e = z.sel(latitude=WB_LATS, longitude=WB_LONS_E)

    # Convert western block to −180/180 convention and concatenate
    z_w = z_w.assign_coords(longitude=z_w.longitude - 360.0)
    z_ea = xr.concat([z_w, z_e], dim="longitude").sortby("longitude")

    print(f"  Domain shape: {dict(z_ea.sizes)}")
    print("  Loading data from GCS ...")
    z_ea = z_ea.load()

    # ── Day-of-year climatology ────────────────────────────────────────────────
    # Standard approach for anomaly computation: group by day-of-year (1–365/366).
    # Note: doy=60 maps to Feb 29 in leap years and Mar 1 in non-leap years —
    # the resulting small misalignment (~1 day around Mar 1) is acceptable and
    # standard practice in meteorology for Z500 anomaly computation.
    print("  Computing day-of-year climatology ...")
    z_clim = (z_ea / G).groupby("time.dayofyear").mean("time")
    z_clim.name = "z500"
    z_clim.attrs = {
        "long_name": "Geopotential height at 500 hPa — day-of-year climatology",
        "units": "gpm",
        "source": f"gs://{WB_URL}",
        "reference_period": f"{REF_START} to {REF_END}",
        "time_of_day": f"{UTC_HOUR:02d} UTC",
    }

    OUT_ZARR.parent.mkdir(parents=True, exist_ok=True)
    z_clim.to_zarr(str(OUT_ZARR))
    print(f"\nSaved → {OUT_ZARR}")
    print(f"Shape : {dict(z_clim.sizes)}")
    print(f"Z500  : {float(z_clim.min()):.0f} – {float(z_clim.max()):.0f} gpm")
