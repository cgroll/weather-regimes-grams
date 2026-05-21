"""
Render an animated GIF of the WeatherBench2 Z500 daily climatology.

One frame per day-of-year (1–366). Saves to output/images/18_z500_climatology.gif.
Run-once: skips if the output already exists.
"""

import io

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image

from wr.paths import ProjPaths

paths = ProjPaths()

OUT_GIF = paths.images_path / "18_z500_climatology.gif"

if OUT_GIF.exists():
    print(f"Already exists, skipping: {OUT_GIF}")
    raise SystemExit(0)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading climatology ...")
ds = xr.open_zarr(str(paths.wb_z500_climatology))
z = ds["z500"].load()   # (dayofyear, longitude, latitude)

lons = z.longitude.values
lats = z.latitude.values
vmin = float(z.min())
vmax = float(z.max())
levels = np.linspace(vmin, vmax, 21)

PROJ  = ccrs.PlateCarree()
TRANS = ccrs.PlateCarree()

# ── Build frames ───────────────────────────────────────────────────────────────
print(f"Rendering {len(z.dayofyear)} frames ...")
frames: list[Image.Image] = []

for i, doy in enumerate(z.dayofyear.values):
    field = z.sel(dayofyear=doy).values.T   # (lat, lon)

    fig, ax = plt.subplots(
        figsize=(9, 5),
        subplot_kw={"projection": PROJ},
    )
    ax.set_extent([-80, 41, 29, 91], crs=TRANS)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")

    cf = ax.contourf(
        lons, lats, field,
        levels=levels,
        cmap="RdBu_r",
        transform=TRANS,
        extend="both",
    )
    ax.contour(
        lons, lats, field,
        levels=levels[::2],
        colors="k",
        linewidths=0.4,
        transform=TRANS,
    )

    plt.colorbar(cf, ax=ax, orientation="vertical", label="Z500 (gpm)", pad=0.02, fraction=0.03)
    ax.set_title(f"Z500 climatology — day {doy:03d}", fontsize=12)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf).copy())

    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(z.dayofyear)}")

# ── Save GIF ───────────────────────────────────────────────────────────────────
OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
print(f"Saving GIF → {OUT_GIF}")
iio.imwrite(
    str(OUT_GIF),
    [np.array(f.convert("RGB")) for f in frames],
    plugin="pillow",
    duration=80,    # ms per frame ≈ ~12 fps
    loop=0,
)
print("Done.")
