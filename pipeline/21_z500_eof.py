# ---
# jupytext:
#   text_representation:
#     format_name: percent
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---

# %% [markdown]
# # Z500 Anomaly EOF Analysis
#
# Empirical Orthogonal Functions (EOFs) of daily ERA5 Z500 anomalies over the
# Euro-Atlantic domain (1979–2021).  Area-weighted PCA identifies the dominant
# recurrent circulation patterns and their time series (Principal Components).
#
# **Method**
# - Grid-point weights: √cos(lat) applied before PCA
# - Randomised SVD (scikit-learn) for efficiency
# - EOF patterns displayed as regression maps in gpm units:
#   *one σ change in PC_k → this many gpm change at each grid point*
# - 20 leading PC time series saved to `data/processed/z500_pcs.parquet`

# %%
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.decomposition import PCA

from wr.paths import ProjPaths

paths = ProjPaths()

IMG_OUT        = paths.images_path / "21_z500_eof_maps.png"
PC_OUT         = paths.z500_pcs
N_DISPLAY      = 9
N_SAVE         = 20

# %% [markdown]
# ## Load Z500 anomalies

# %%
print("Loading ERA5 Z500 …")
z500_ds = xr.open_zarr(str(paths.era5_z500_daily_zarr))["z500"].load()

print("Loading WB2 climatology …")
clim = xr.open_zarr(str(paths.wb_z500_climatology))["z500"].load()

doy = z500_ds.time.dt.dayofyear
z500_anom = (z500_ds - clim.sel(dayofyear=doy))

z500_dates = pd.to_datetime(z500_anom.time.values).normalize()
z500_anom  = z500_anom.assign_coords(time=z500_dates)
z500_anom  = z500_anom.sel(time=slice("1979-01-01", "2021-12-31"))

lons = z500_anom.longitude.values   # (n_lon,)  — first spatial dim
lats = z500_anom.latitude.values    # (n_lat,)  — second spatial dim
n_lon, n_lat = len(lons), len(lats)
T = len(z500_anom.time)

print(f"Shape: {z500_anom.shape}  (time, lon, lat)")
print(f"Domain: lon {lons[0]:.1f}–{lons[-1]:.1f}, lat {lats[0]:.1f}–{lats[-1]:.1f}")

# %% [markdown]
# ## Area-weighted PCA

# %%
# Data matrix (T, n_lon * n_lat)  — flatten lon×lat together
X = z500_anom.values.reshape(T, n_lon * n_lat).astype(np.float32)

# Area weights: sqrt(cos(lat)) tiled across longitudes
# After reshape, index = i_lon * n_lat + i_lat → lat changes fastest
lat_w = np.sqrt(np.cos(np.deg2rad(lats)))           # (n_lat,)
W = np.tile(lat_w[None, :], (n_lon, 1)).flatten()   # (n_lon * n_lat,)

X_w = X * W[None, :]   # weighted data matrix  (T, n_grid)
print(f"Weighted matrix: {X_w.shape}, running PCA for {N_SAVE} components …")

pca = PCA(n_components=N_SAVE, svd_solver="randomized", random_state=42)
pc_scores = pca.fit_transform(X_w)    # (T, N_SAVE)

var_frac = pca.explained_variance_ratio_    # fraction per EOF
cum_var  = np.cumsum(var_frac)

print(f"Variance: " + "  ".join(
    f"EOF{k+1}={var_frac[k]*100:.1f}%" for k in range(5)
))
print(f"Cumulative: EOF9={cum_var[8]*100:.1f}%  EOF20={cum_var[19]*100:.1f}%")

# EOF patterns in physical space (gpm units: 1-σ PC amplitude)
# components_ has unit norm in weighted space; divide by W → physical space;
# multiply by sqrt(eigenvalue) → gpm per 1 std of PC
eofs_flat  = pca.components_ / W[None, :]                          # (N_SAVE, n_grid)
eofs_gpm   = eofs_flat * np.sqrt(pca.explained_variance_[:, None]) # (N_SAVE, n_grid)
eofs_maps  = eofs_gpm.reshape(N_SAVE, n_lon, n_lat)                # (N_SAVE, n_lon, n_lat)

# Flip sign so the largest-amplitude grid point in each EOF is positive
for k in range(N_SAVE):
    if eofs_maps[k].min() + eofs_maps[k].max() < 0:
        eofs_maps[k]  *= -1
        pc_scores[:, k] *= -1

# %% [markdown]
# ## Save PC time series

# %%
pc_df = pd.DataFrame(
    pc_scores,
    index=pd.DatetimeIndex(z500_anom.time.values, name="date"),
    columns=[f"PC{k+1:02d}" for k in range(N_SAVE)],
)
PC_OUT.parent.mkdir(parents=True, exist_ok=True)
pc_df.to_parquet(PC_OUT)
print(f"PC time series saved → {PC_OUT}  (shape {pc_df.shape})")
print(pc_df.describe().loc[["mean", "std"]].to_string())

# %% [markdown]
# ## Combined figure: EOF maps + variance explained

# %%
PROJ  = ccrs.PlateCarree()
TRANS = ccrs.PlateCarree()
EXTENT = [-80, 41, 29, 91]

fig = plt.figure(figsize=(18, 17))
gs = gridspec.GridSpec(
    4, 3, figure=fig,
    height_ratios=[1, 1, 1, 0.45],
    hspace=0.35, wspace=0.05,
)

# ── EOF maps (top 3 × 3) ──────────────────────────────────────────────────────
for k in range(N_DISPLAY):
    ax = fig.add_subplot(gs[k // 3, k % 3], projection=PROJ)
    eof = eofs_maps[k]   # (n_lon, n_lat)

    vmax = np.percentile(np.abs(eof), 99)
    levels = np.linspace(-vmax, vmax, 21)

    ax.set_extent(EXTENT, crs=TRANS)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle=":")
    ax.add_feature(cfeature.LAND,      facecolor="#f0f0f0", zorder=0)

    cf = ax.contourf(
        lons, lats, eof.T,
        levels=levels, cmap="RdBu_r", transform=TRANS, extend="both",
    )
    ax.contour(
        lons, lats, eof.T,
        levels=levels[::4], colors="k", linewidths=0.35, transform=TRANS,
    )
    ax.contour(
        lons, lats, eof.T,
        levels=[0], colors="k", linewidths=0.8, transform=TRANS,
    )
    plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.02, fraction=0.035,
                 label="gpm" if k % 3 == 2 else "")

    ax.set_title(
        f"EOF {k+1}  ({var_frac[k]*100:.1f}%,  cum. {cum_var[k]*100:.1f}%)",
        fontsize=10, pad=4,
    )
    ax.gridlines(linewidth=0.2, color="grey", alpha=0.5)

# ── Variance plot (bottom, full width) ───────────────────────────────────────
var_ax = fig.add_subplot(gs[3, :])
eof_nums = np.arange(1, N_SAVE + 1)

bars = var_ax.bar(eof_nums, var_frac * 100, color="steelblue", alpha=0.7,
                  label="Individual")
ax2  = var_ax.twinx()
ax2.plot(eof_nums, cum_var * 100, color="tomato", lw=2, marker="o",
         markersize=4, label="Cumulative")
ax2.axhline(50, color="tomato", lw=0.7, ls="--", alpha=0.6)
ax2.axhline(75, color="tomato", lw=0.7, ls="--", alpha=0.6)
for pct in [50, 75]:
    ax2.annotate(f"{pct}%", xy=(N_SAVE + 0.3, pct), va="center",
                 color="tomato", fontsize=8)

var_ax.set_xlabel("EOF number", fontsize=11)
var_ax.set_ylabel("Variance explained (%)", fontsize=11, color="steelblue")
var_ax.tick_params(axis="y", labelcolor="steelblue")
ax2.set_ylabel("Cumulative variance (%)", fontsize=11, color="tomato")
ax2.tick_params(axis="y", labelcolor="tomato")
ax2.set_ylim(0, 105)
var_ax.set_xticks(eof_nums)
var_ax.set_title("Variance explained by leading Z500 EOFs", fontsize=11)

lines1, labels1 = var_ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
var_ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")

fig.suptitle(
    "ERA5 Z500 anomaly EOFs — Euro-Atlantic (1979–2021, area-weighted PCA)",
    fontsize=13, y=1.005,
)

fig.savefig(IMG_OUT, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {IMG_OUT}")

# %% [markdown]
# ```{figure} ../../output/images/21_z500_eof_maps.png
# :name: fig-21-z500-eof-maps
# Top nine area-weighted EOFs of daily ERA5 Z500 anomalies (1979–2021).
# Filled contours show the regression pattern in gpm (one σ change in the PC).
# Black contour lines are added at every fourth level; the zero line is bold.
# Bottom panel: individual (blue bars) and cumulative (red line) variance
# explained by the leading 20 EOFs.
# ```
