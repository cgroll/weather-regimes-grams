"""
Download the WeatherBench2 ERA5 Z500 climatology at 6-hourly resolution on the
Grams 0.5° EOF domain.

Pure data script — no visualizations.

Purpose
-------
Script 06 forms Z500 anomalies with the *year-round* Grams mean, which leaves
the annual cycle in the anomaly.  This file provides a seasonally varying
(day-of-year × hour-of-day) climatology so that the effect of the missing
annual cycle on the recomputed IWR can be tested (script 25).

Source
------
WeatherBench2 ERA5 hourly climatology at 0.25° (1440 × 721 global grid)
gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr

Reference period : 1990–2019  (Grams uses 1979–2019)
Hours            : 00, 06, 12, 18 UTC
Chunks           : (3 hours, 3 days, 1 level, global) — only the 500 hPa
                   chunks are read, ≈ 2.9 GB transfer for the full year.

Domain
------
Grams EOF domain: 30–90 °N, 80 °W–40 °E.  Every second 0.25° point is
selected, which gives exactly the 0.5° grid of the Grams patterns
(121 × 241 points).

Output
------
data/downloads/wb/z500_climatology_6h_0p5deg.zarr
  variable  z500        geopotential height at 500 hPa (gpm)
  dims      hour × dayofyear × latitude × longitude   (4 × 366 × 121 × 241)
  latitude  30 → 90 (S→N),  longitude −80 → 40
"""

import numpy as np
import xarray as xr
import gcsfs

from wr.paths import ProjPaths

paths = ProjPaths()
OUT_ZARR = paths.wb_z500_climatology_6h

WB_CLIM_URL = (
    "weatherbench2/datasets/era5-hourly-climatology/"
    "1990-2019_6h_1440x721.zarr"
)

G = 9.80665   # m/s²

LAT_MIN, LAT_MAX = 30.0, 90.0   # °N
LON_MIN, LON_MAX = -80.0, 40.0  # °E

# ── Cache check ────────────────────────────────────────────────────────────────
if OUT_ZARR.exists():
    print(f"Using cached climatology: {OUT_ZARR}")
else:
    print(f"Opening WeatherBench2 climatology: gs://{WB_CLIM_URL}")
    fs    = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(WB_CLIM_URL)
    ds    = xr.open_zarr(store, consolidated=True)

    print(f"  Lat  : {float(ds.latitude[0]):.2f} → {float(ds.latitude[-1]):.2f}")
    print(f"  Lon  : {float(ds.longitude[0]):.2f} → {float(ds.longitude[-1]):.2f}")
    print(f"  Hours: {ds.hour.values}")

    z = ds["geopotential"].sel(level=500)

    # ── Select EOF domain at 0.5° ──────────────────────────────────────────────
    # WB2 uses 0–360 longitude; select the exact 0.5° points by value.
    lats   = np.arange(LAT_MIN, LAT_MAX + 0.25, 0.5)
    lons_w = np.arange(360.0 + LON_MIN, 360.0, 0.5)   # 280.0 … 359.5
    lons_e = np.arange(0.0, LON_MAX + 0.25, 0.5)      #   0.0 …  40.0

    z_w = z.sel(latitude=lats, longitude=lons_w)
    z_e = z.sel(latitude=lats, longitude=lons_e)
    z_w = z_w.assign_coords(longitude=z_w.longitude - 360.0)
    z_ea = xr.concat([z_w, z_e], dim="longitude").sortby("longitude").sortby("latitude")
    z_ea = z_ea.transpose("hour", "dayofyear", "latitude", "longitude")

    assert z_ea.sizes["latitude"] == 121 and z_ea.sizes["longitude"] == 241, dict(z_ea.sizes)
    print(f"  Domain shape: {dict(z_ea.sizes)}")
    print("  Loading from GCS (≈ 2.9 GB transfer) ...")
    z_ea = z_ea.load()

    z_clim = (z_ea / G).rename("z500").drop_vars("level")
    z_clim.attrs = {
        "long_name": "Geopotential height at 500 hPa — day-of-year × hour climatology",
        "units": "gpm",
        "source": f"gs://{WB_CLIM_URL}",
        "reference_period": "1990-2019",
    }

    OUT_ZARR.parent.mkdir(parents=True, exist_ok=True)
    z_clim.to_dataset().to_zarr(str(OUT_ZARR), zarr_format=2)
    print(f"\nSaved → {OUT_ZARR}")
    print(f"Shape : {dict(z_clim.sizes)}")
    print(f"Z500  : {float(z_clim.min()):.0f} – {float(z_clim.max()):.0f} gpm")
