"""
Download ERA5 Z500 daily climatology from WeatherBench2 (1.5° global grid).

WeatherBench2 provides a pre-computed hourly climatology zarr — no need to
download 41 years of ERA5 and average manually.  We simply select hour=12,
level=500, and the Euro-Atlantic domain.

Source
------
WeatherBench2 ERA5 hourly climatology at 1.5° (240 × 121 global grid)
gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121.zarr

Reference period : 1990–2019  (WeatherBench standard; close to Grams 1979–2019)

Domain
------
Grams WR EOF domain (nearest 1.5° grid points):
  latitude  : 30.0 – 90.0°N   41 points  (1.5° step)
  longitude : −79.5 – 40.5°E  81 points  (1.5° step)

Time
----
hour = 12 UTC  =  13:00 CET (UTC+1, no DST) — matches the actual data script.

Output
------
data/downloads/wb/z500_climatology.zarr
  variable  z500        geopotential height at 500 hPa (gpm)
  dims      dayofyear × latitude × longitude
"""

import numpy as np
import xarray as xr
import gcsfs

from wr.paths import ProjPaths

paths = ProjPaths()
OUT_ZARR = paths.wb_z500_climatology

WB_CLIM_URL = (
    "weatherbench2/datasets/era5-hourly-climatology/"
    "1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr"
)

G        = 9.80665   # m/s²
UTC_HOUR = 12        # 12 UTC = 13:00 CET

# Euro-Atlantic domain on the WeatherBench 1.5° grid (0–360 lon convention).
# 280.5° = −79.5°E,  40.5° = 40.5°E — nearest 1.5° points to the Grams −80/40 domain.
WB_LATS   = np.arange(30.0, 91.5, 1.5)
WB_LONS_W = np.arange(280.5, 360.0, 1.5)   # western block: −79.5 to −1.5
WB_LONS_E = np.arange(0.0, 42.0, 1.5)      # eastern block:   0.0 to 40.5

# ── Cache check ────────────────────────────────────────────────────────────────
if OUT_ZARR.exists():
    print(f"Using cached climatology: {OUT_ZARR}")
else:
    print(f"Opening WeatherBench2 climatology: gs://{WB_CLIM_URL}")
    fs    = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(WB_CLIM_URL)
    ds    = xr.open_zarr(store, consolidated=True)

    print(f"  Variables : {list(ds.data_vars)}")
    print(f"  Coords    : {list(ds.coords)}")
    print(f"  Lat       : {float(ds.latitude[0]):.1f} → {float(ds.latitude[-1]):.1f}")
    print(f"  Lon       : {float(ds.longitude[0]):.1f} → {float(ds.longitude[-1]):.1f}")

    # ── Select geopotential at 500 hPa, 12 UTC ────────────────────────────────
    z = ds["geopotential"].sel(level=500, hour=UTC_HOUR)

    # ── Select Euro-Atlantic domain ────────────────────────────────────────────
    z_w = z.sel(latitude=WB_LATS, longitude=WB_LONS_W, method="nearest")
    z_e = z.sel(latitude=WB_LATS, longitude=WB_LONS_E, method="nearest")

    z_w = z_w.assign_coords(longitude=z_w.longitude - 360.0)
    z_ea = xr.concat([z_w, z_e], dim="longitude").sortby("longitude")

    print(f"  Domain shape: {dict(z_ea.sizes)}")
    print("  Loading from GCS ...")
    z_ea = z_ea.load()

    z_clim = (z_ea / G).rename("z500")
    z_clim.attrs = {
        "long_name": "Geopotential height at 500 hPa — day-of-year climatology",
        "units": "gpm",
        "source": f"gs://{WB_CLIM_URL}",
        "reference_period": "1990-2019",
        "time_of_day": f"{UTC_HOUR:02d} UTC",
    }

    OUT_ZARR.parent.mkdir(parents=True, exist_ok=True)
    z_clim.to_zarr(str(OUT_ZARR), zarr_format=2)
    print(f"\nSaved → {OUT_ZARR}")
    print(f"Shape : {dict(z_clim.sizes)}")
    print(f"Z500  : {float(z_clim.min()):.0f} – {float(z_clim.max()):.0f} gpm")
