"""
Render two animated GIFs for a given year: raw Z500 and Z500 anomaly.

Anomaly = ERA5 daily Z500 (12 UTC) minus the WeatherBench2 climatology
matched by day-of-year.

Usage
-----
    uv run python pipeline/19_z500_anomaly_gif.py [YEAR]   (default: 2019)

Output
------
    output/images/19_z500_{year}.gif
    output/images/19_z500_anomaly_{year}.gif
"""

import io
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image

from wr.paths import ProjPaths

# ── Args ───────────────────────────────────────────────────────────────────────
year = int(sys.argv[1]) if len(sys.argv) > 1 else 2020

paths = ProjPaths()
OUT_RAW  = paths.images_path / f"19_z500_{year}.gif"
OUT_ANOM = paths.images_path / f"19_z500_anomaly_{year}.gif"

if OUT_RAW.exists() and OUT_ANOM.exists():
    print(f"Both GIFs already exist, skipping.")
    raise SystemExit(0)

# ── Load data ──────────────────────────────────────────────────────────────────
print(f"Loading ERA5 Z500 for {year} ...")
era5 = xr.open_zarr(str(paths.era5_z500_year_zarr(year)))["z500"].load()

print("Loading climatology ...")
clim = xr.open_zarr(str(paths.wb_z500_climatology))["z500"].load()

doy = era5.time.dt.dayofyear
clim_aligned = clim.sel(dayofyear=doy)
anomaly = (era5 - clim_aligned).rename("z500_anomaly")

lons = era5.longitude.values
lats = era5.latitude.values

raw_levels  = np.linspace(float(clim.min()),  float(clim.max()),  21)
amax        = max(float(np.abs(anomaly).quantile(0.99)), 1.0)
anom_levels = np.linspace(-amax, amax, 21)

PROJ  = ccrs.PlateCarree()
TRANS = ccrs.PlateCarree()

# ── Render helper ──────────────────────────────────────────────────────────────
def render_frame(field, levels, cmap, cbar_label, title) -> Image.Image:
    fig, ax = plt.subplots(figsize=(9, 5), subplot_kw={"projection": PROJ})
    ax.set_extent([-80, 41, 29, 91], crs=TRANS)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
    cf = ax.contourf(lons, lats, field, levels=levels, cmap=cmap, transform=TRANS, extend="both")
    ax.contour(lons, lats, field, levels=levels[::2], colors="k", linewidths=0.4, transform=TRANS)
    plt.colorbar(cf, ax=ax, orientation="vertical", label=cbar_label, pad=0.02, fraction=0.03)
    ax.set_title(title, fontsize=12)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()

# ── Build frames ───────────────────────────────────────────────────────────────
n = len(era5.time)
raw_frames: list[Image.Image] = []
anom_frames: list[Image.Image] = []

print(f"Rendering {n} frames ...")
for i, t in enumerate(era5.time.values):
    date_str = str(t)[:10]
    raw_frames.append(render_frame(
        era5.isel(time=i).values.T,
        raw_levels, "RdBu_r",
        "Z500 (gpm)", f"Z500 — {date_str}",
    ))
    anom_frames.append(render_frame(
        anomaly.isel(time=i).values.T,
        anom_levels, "RdBu_r",
        "Z500 anomaly (gpm)", f"Z500 anomaly — {date_str}",
    ))
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{n}")

# ── Save GIFs ──────────────────────────────────────────────────────────────────
OUT_RAW.parent.mkdir(parents=True, exist_ok=True)

for out, frames in [(OUT_RAW, raw_frames), (OUT_ANOM, anom_frames)]:
    if not out.exists():
        print(f"Saving → {out}")
        iio.imwrite(
            str(out),
            [np.array(f.convert("RGB")) for f in frames],
            plugin="pillow",
            duration=80,
            loop=0,
        )
print("Done.")
