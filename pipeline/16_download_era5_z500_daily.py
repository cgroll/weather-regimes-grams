"""
Download ERA5 Z500 daily snapshots (2024–2025) from ARCO ERA5 on GCP.

Produces data on the same 1.5° grid as the WeatherBench2 climatology
(pipeline/15_download_wb_z500_climatology.py) so that anomalies can be
computed by a simple subtraction.

Source
------
ARCO ERA5 hourly 0.25° global
gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3

Grid alignment
--------------
WeatherBench 1.5° grid points lie exactly on the ARCO 0.25° grid because
1.5 / 0.25 = 6 (integer).  This script therefore uses exact coordinate
selection (no interpolation) to extract the WeatherBench grid points:

  latitude  : 30.0 – 90.0°N   every 1.5°  (41 pts)
  longitude : −79.5 – 40.5°E  every 1.5°  (81 pts)

Time
----
12 UTC daily  =  13:00 CET (UTC+1, no DST) — same hour as the climatology.
Period        : 2024-01-01 – 2025-12-31

Output
------
data/downloads/era5/z500_euro_atlantic_2024_2025.zarr
  variable  z500        geopotential height at 500 hPa (gpm)
  dims      time × latitude × longitude
"""

import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

from wr.paths import ProjPaths

paths = ProjPaths()
OUT_ZARR = paths.era5_z500_daily_zarr

ARCO_URL  = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
G         = 9.80665   # m/s²
UTC_HOUR  = 12        # must match climatology script

START     = "2024-01-01"
END       = "2025-12-31"

# Euro-Atlantic domain — exactly the WeatherBench 1.5° grid points.
# ARCO uses 0–360 longitude and descending latitude (90 → −90).
WB_LATS   = np.arange(30.0, 91.5, 1.5)         # ascending (for output)
WB_LONS_W = np.arange(280.5, 360.0, 1.5)       # −79.5 to −1.5  (0–360)
WB_LONS_E = np.arange(0.0, 42.0, 1.5)          # 0 to 40.5      (0–360)

# ── Cache check ────────────────────────────────────────────────────────────────
if OUT_ZARR.exists():
    print(f"Using cached data: {OUT_ZARR}")
else:
    print(f"Opening ARCO ERA5: gs://{ARCO_URL}")
    fs    = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(ARCO_URL)
    ds    = xr.open_zarr(store, consolidated=True)

    lat0  = float(ds.latitude[0])
    lat1  = float(ds.latitude[-1])
    lon0  = float(ds.longitude[0])
    lon1  = float(ds.longitude[-1])
    print(f"  Lat : {lat0:.2f} → {lat1:.2f}  ({len(ds.latitude)} pts)")
    print(f"  Lon : {lon0:.2f} → {lon1:.2f}  ({len(ds.longitude)} pts)")
    print(f"  Time: {str(ds.time.values[0])[:13]} → {str(ds.time.values[-1])[:13]}")

    # ── Select 12 UTC daily snapshots ─────────────────────────────────────────
    times_12utc = pd.date_range(
        f"{START} {UTC_HOUR:02d}:00",
        f"{END}   {UTC_HOUR:02d}:00",
        freq="24h",
    )
    print(f"\n  Selecting {len(times_12utc)} daily 12 UTC time steps ({START}–{END}) ...")

    z = ds["geopotential"].sel(level=500, time=times_12utc)

    # ── Select Euro-Atlantic domain on the 1.5° WeatherBench grid ─────────────
    # ARCO latitude is descending → reverse WB_LATS for the selection, then
    # sortby at the end restores ascending order.
    arco_lats = WB_LATS[::-1]   # 90, 88.5, ..., 30  (descending, matching ARCO)

    z_w = z.sel(latitude=arco_lats, longitude=WB_LONS_W)
    z_e = z.sel(latitude=arco_lats, longitude=WB_LONS_E)

    # Convert western block to −180/180 and concatenate
    z_w = z_w.assign_coords(longitude=z_w.longitude - 360.0)
    z_ea = (
        xr.concat([z_w, z_e], dim="longitude")
        .sortby("longitude")
        .sortby("latitude")    # ascending south → north
    )

    # Verify exact 1.5° spacing
    dlat = abs(float(z_ea.latitude[1]) - float(z_ea.latitude[0]))
    dlon = abs(float(z_ea.longitude[1]) - float(z_ea.longitude[0]))
    assert abs(dlat - 1.5) < 1e-4, f"Unexpected lat spacing: {dlat}"
    assert abs(dlon - 1.5) < 1e-4, f"Unexpected lon spacing: {dlon}"
    print(f"  Domain shape: {dict(z_ea.sizes)}  ({dlat:.1f}° × {dlon:.1f}°)")

    # ── Convert and save ───────────────────────────────────────────────────────
    print("  Loading data from GCS ...")
    z500 = (z_ea / G).rename("z500")
    z500.attrs = {
        "long_name": "Geopotential height at 500 hPa",
        "units": "gpm",
        "source": f"gs://{ARCO_URL}",
        "note": (
            "Raw field — not yet an anomaly. "
            "Subtract day-of-year climatology from "
            "data/downloads/wb/z500_climatology.zarr to compute anomalies."
        ),
    }

    z500 = z500.load()
    OUT_ZARR.parent.mkdir(parents=True, exist_ok=True)
    z500.to_zarr(str(OUT_ZARR))
    print(f"\nSaved → {OUT_ZARR}")
    print(f"Shape : {dict(z500.sizes)}")
    print(f"Z500  : {float(z500.min()):.0f} – {float(z500.max()):.0f} gpm")
    print(f"Time  : {str(z500.time.values[0])[:13]} → {str(z500.time.values[-1])[:13]}")
