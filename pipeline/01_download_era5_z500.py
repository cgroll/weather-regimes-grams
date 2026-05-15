"""Download ERA5 Z0500 for weather regime projection from ARCO ERA5 on GCP.

Pure data script — no visualizations.

┌─────────────────────────────────────────────────────────────────────────────┐
│  Why 3-hourly?                                                              │
│  The Grams WR dataset uses 3-hourly timesteps (00, 03, 06, 09, 12, 15,     │
│  18, 21 UTC).  ERA5 is available hourly; we select every third hour.       │
│                                                                             │
│  How much padding for the Lanczos filter?                                   │
│  The 10-day LP filter uses a 161-point Lanczos kernel at Δt = 3 h.         │
│  This requires (161 − 1) / 2 = 80 guard timesteps on each side             │
│  = 80 × 3 h = 240 h = 10 days minimum.  PADDING_DAYS = 20 gives a         │
│  comfortable safety margin so filter edge effects are negligible at the    │
│  boundaries of the analysis window.                                         │
│                                                                             │
│  Data source                                                                │
│  ARCO ERA5: gs://gcp-public-data-arco-era5/  (public, no auth needed)     │
│  0.25° global, hourly → we regrid to 0.5° by selecting every other point. │
└─────────────────────────────────────────────────────────────────────────────┘

Output: data/downloads/era5/z0500_<start>_<end>.nc
  - variable  : z0500   geopotential height at 500 hPa (m), raw (not anomaly)
  - time      : 3-hourly, analysis period ± PADDING_DAYS days
  - latitude  : 30–90 °N, 0.5° (121 points)
  - longitude : −80–40 °E, 0.5° (241 points)
"""

import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

from wr.paths import ProjPaths

paths = ProjPaths()

# ── Configuration ──────────────────────────────────────────────────────────────
ANALYSIS_START = "2024-11-01"    # desired analysis period (no padding)
ANALYSIS_END   = "2025-03-31"

# 20 days per side: theoretical minimum is 10, 20 gives comfortable margin
PADDING_DAYS = 20

# EOF domain — must match the Grams projection domain exactly.
# This is the region covered by Normed_Z0500-patterns_EOFdomain.nc
# (121 lat pts × 241 lon pts at 0.5°).
LAT_MIN, LAT_MAX = 30.0, 90.0   # °N
LON_MIN, LON_MAX = -80.0, 40.0  # °E

# ARCO ERA5 zarr store (public, anonymous access)
ARCO_URL = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# ── Derived download window ────────────────────────────────────────────────────
dl_start = pd.Timestamp(ANALYSIS_START) - pd.Timedelta(days=PADDING_DAYS)
dl_end   = pd.Timestamp(ANALYSIS_END)   + pd.Timedelta(days=PADDING_DAYS)

print(f"Analysis period : {ANALYSIS_START} → {ANALYSIS_END}")
print(f"Download window : {dl_start.date()} → {dl_end.date()}  (±{PADDING_DAYS} d padding)")

# 3-hourly timestamps for the full download window
times_3h = pd.date_range(dl_start, dl_end, freq="3h")
print(f"Timesteps       : {len(times_3h)}  ({times_3h[0]} … {times_3h[-1]})")

# ── Open ARCO ERA5 lazily ──────────────────────────────────────────────────────
print(f"\nOpening gs://{ARCO_URL} ...")
fs    = gcsfs.GCSFileSystem(token="anon")
store = fs.get_mapper(ARCO_URL)
ds    = xr.open_zarr(store, consolidated=True)

lat0, lat1 = float(ds.latitude[0]), float(ds.latitude[-1])
lon0, lon1 = float(ds.longitude[0]), float(ds.longitude[-1])
dlat = abs(lat1 - lat0) / (len(ds.latitude) - 1)
dlon = abs(lon1 - lon0) / (len(ds.longitude) - 1)
print(f"Lat  : {lat0:.2f} → {lat1:.2f}  ({len(ds.latitude)} pts, {dlat:.2f}°)")
print(f"Lon  : {lon0:.2f} → {lon1:.2f}  ({len(ds.longitude)} pts, {dlon:.2f}°)")

# ── Select time, level, and spatial domain ────────────────────────────────────
# Select 500 hPa geopotential at 3-hourly timesteps
z500 = ds["geopotential"].sel(level=500, time=times_3h)

# ARCO ERA5 uses 0–360 longitude; convert EOF domain to that convention
# lon -80..0 → 280..360,  lon 0..40 stays 0..40
z500_west = z500.sel(
    latitude=slice(LAT_MAX, LAT_MIN),      # N→S stored order
    longitude=slice(360.0 + LON_MIN, 360.0),
)
z500_east = z500.sel(
    latitude=slice(LAT_MAX, LAT_MIN),
    longitude=slice(0.0, LON_MAX),
)

# Shift western block to negative longitudes and concatenate
z500_west = z500_west.assign_coords(longitude=z500_west.longitude - 360.0)
z500_domain = xr.concat([z500_west, z500_east], dim="longitude").sortby("longitude")

# ── Regrid 0.25° → 0.5° ───────────────────────────────────────────────────────
# Both grids are regular with ARCO lat starting at 90.0 and lon at multiples
# of 0.25°.  Selecting every second point gives the exact 0.5° grid.
z500_half = z500_domain.isel(
    latitude=slice(None, None, 2),
    longitude=slice(None, None, 2),
)

# Verify grid spacing
dlat = abs(float(z500_half.latitude[1]) - float(z500_half.latitude[0]))
dlon = abs(float(z500_half.longitude[1]) - float(z500_half.longitude[0]))
assert abs(dlat - 0.5) < 1e-6, f"Unexpected lat spacing: {dlat}"
assert abs(dlon - 0.5) < 1e-6, f"Unexpected lon spacing: {dlon}"
print(f"\nGrid after regrid: {dict(z500_half.sizes)}  ({dlat:.1f}° × {dlon:.1f}°)")

# ── Convert geopotential (m²/s²) → geopotential height (m) ───────────────────
z0500 = (z500_half / 9.80665).rename("z0500")
z0500.attrs = {
    "long_name": "Geopotential height at 500 hPa",
    "units": "m",
    "note": (
        "Raw field — not yet an anomaly, not yet LP-filtered. "
        "Subtract 1979-2019 climatology and apply 161-pt Lanczos LP filter "
        "before computing WR projections."
    ),
}

# ── Compute and save ───────────────────────────────────────────────────────────
out_path = paths.era5_z500_nc(ANALYSIS_START, ANALYSIS_END)
out_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Downloading and writing → {out_path}")
print("(This triggers the actual data transfer from GCS — may take a few minutes.)")

z0500.load().to_netcdf(out_path)

size_mb = out_path.stat().st_size / 1e6
print(f"\nSaved {out_path.name}  ({size_mb:.0f} MB)")
print(f"Shape: {dict(z0500.sizes)}")
print(f"Time range: {str(z0500.time.values[0])[:16]} → {str(z0500.time.values[-1])[:16]}")
