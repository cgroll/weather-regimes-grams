"""Download ERA5 Z0500 around 2025-06-01 00 UTC from ARCO ERA5 on GCP.

Pure data script — no visualizations.

Purpose
-------
The Grams Zenodo package ships one pre-processed field,
`Z0500_20250601_00.nc` (10-day Lanczos LP-filtered Z500 anomaly).  To
recompute that field from raw ERA5 (script 27) the 161-point Lanczos filter
needs 80 timesteps (10 days at Δt = 3 h) on each side of the target time.
This script downloads that window plus 2 days of margin.

Same source, domain and regridding as 01_download_era5_z500.py:
ARCO ERA5 0.25° hourly → 3-hourly, EOF domain 30–90 °N, 80 °W–40 °E,
every second point → 0.5° (121 × 241).

Why raw chunk access instead of xarray?
---------------------------------------
Every ARCO chunk holds one global hour on all 37 levels (≈ 80 MB
compressed), so ≈ 190 timesteps transfer ≈ 15 GB for ≈ 20 MB of output.
Loading that through xarray gives no progress output and loses everything
on an interruption.  Here each chunk is fetched directly (several in
parallel), decoded, cropped to 500 hPa + EOF domain and cached as one small
.npy file per timestep, so an interrupted run resumes where it stopped.

Output: data/downloads/era5/z0500_20250601_20250601.nc
Cache : data/downloads/era5/golden_steps/<YYYYmmddHH>.npy
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import gcsfs
import numcodecs
import numpy as np
import pandas as pd
import xarray as xr

from wr.paths import ProjPaths

paths = ProjPaths()

TARGET       = "2025-06-01"
PADDING_DAYS = 12   # 10 days needed by the filter + 2 days margin
N_WORKERS    = 8

ARCO_URL  = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
VAR       = "geopotential"
TIME_ORIG = pd.Timestamp("1900-01-01")   # ARCO time axis: hours since 1900-01-01

# Index selection on the 0.25° global grid (lat 90 → −90, lon 0 → 359.75)
LEVEL_IDX = 21                         # 500 hPa (asserted below)
LAT_IDX   = np.arange(0, 241, 2)       # 90.0 → 30.0 °N, 0.5°
LON_IDX   = np.concatenate([np.arange(1120, 1440, 2),   # 280.0 … 359.5 → −80.0 … −0.5
                            np.arange(0, 161, 2)])      #   0.0 …  40.0
LATS = 90.0 - 0.25 * LAT_IDX
LONS = np.where(LON_IDX >= 720, LON_IDX * 0.25 - 360.0, LON_IDX * 0.25)

out_path  = paths.era5_z500_nc(TARGET, TARGET)
cache_dir = paths.era5_path / "golden_steps"

if out_path.exists():
    print(f"Using cached file: {out_path}")
else:
    times = pd.date_range(
        pd.Timestamp(TARGET) - pd.Timedelta(days=PADDING_DAYS),
        pd.Timestamp(TARGET) + pd.Timedelta(days=PADDING_DAYS),
        freq="3h",
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    fs   = gcsfs.GCSFileSystem(token="anon")
    meta = json.loads(fs.cat(f"{ARCO_URL}/{VAR}/.zarray"))
    assert meta["chunks"] == [1, 37, 721, 1440], meta["chunks"]
    levels = json.loads(fs.cat(f"{ARCO_URL}/level/.zarray"))   # sanity: 37 levels
    assert levels["shape"] == [37]
    codec = numcodecs.get_codec(meta["compressor"])
    shape = tuple(meta["chunks"])

    def _cache_file(ts: pd.Timestamp):
        return cache_dir / f"{ts:%Y%m%d%H}.npy"

    def _fetch(ts: pd.Timestamp) -> pd.Timestamp:
        t_idx = int((ts - TIME_ORIG) / pd.Timedelta("1h"))
        raw   = fs.cat(f"{ARCO_URL}/{VAR}/{t_idx}.0.0.0")
        arr   = np.frombuffer(codec.decode(raw), dtype=meta["dtype"]).reshape(shape)
        field = arr[0, LEVEL_IDX][np.ix_(LAT_IDX, LON_IDX)] / 9.80665
        np.save(_cache_file(ts), field.astype(np.float32))
        return ts

    todo = [t for t in times if not _cache_file(t).exists()]
    print(f"Timesteps: {len(times)}  ({times[0]} … {times[-1]}),  "
          f"cached: {len(times) - len(todo)},  to fetch: {len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(_fetch, t) for t in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            print(f"  [{n:3d}/{len(todo)}] {fut.result():%Y-%m-%d %H} UTC", flush=True)

    # Sanity check on 500 hPa: Z500 heights lie roughly within 4800–6000 m
    data = np.stack([np.load(_cache_file(t)) for t in times])
    assert 4500 < data.min() and data.max() < 6200, (data.min(), data.max())

    z0500 = xr.DataArray(
        data, dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": LATS, "longitude": LONS},
        name="z0500",
        attrs={
            "long_name": "Geopotential height at 500 hPa",
            "units": "m",
            "note": "Raw field — not yet an anomaly, not yet LP-filtered.",
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z0500.to_netcdf(out_path)
    print(f"Saved {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)  "
          f"Z500 {data.min():.0f} – {data.max():.0f} m")
