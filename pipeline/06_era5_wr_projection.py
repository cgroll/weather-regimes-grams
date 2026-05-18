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
# # ERA5-based Weather Regime Projections
#
# Recomputes IWR projections from raw ERA5 Z500 and compares against the
# pre-computed Grams reference values for the analysis period 2024-11-01 –
# 2025-03-31.
#
# ## Pipeline
#
# 1. Load ERA5 Z500 (downloaded with ±20-day padding for the Lanczos filter).
# 2. Subtract the 1979–2019 year-round climatological mean to form anomalies.
# 3. Apply 161-point Lanczos low-pass filter (cutoff 240 h = 10 days, Δt = 3 h).
# 4. Crop to the analysis period.
# 5. Normalise each timestep by the seasonal weight `normwgt`.
# 6. Project onto the 7 normalised regime patterns.
# 7. Standardise (subtract 1979–2019 mean, divide by std) → IWR.
# 8. Compare with pre-computed `WRI_projections.csv`.
#
# ## Climatology note
#
# The Grams dataset provides only the **year-round** mean
# (`CLIM_Z@500_year_1979-2019.nc`), not a seasonal daily climatology.
# Subtracting the year-round mean leaves the annual cycle in the anomaly.
# That cycle survives the 10-day LP filter (period 365 d ≫ 10 d), so the
# LP-filtered anomaly here still contains seasonal variation.  The seasonal
# `normwgt` division in step 5 compensates for the seasonal amplitude; even so,
# a grid-point-wise daily climatology would give a cleaner anomaly and tighter
# agreement with the pre-computed IWR.

# %%
import calendar

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import convolve1d

from wr.paths import ProjPaths
from wr.regimes import WR_NAMES

paths = ProjPaths()

ANALYSIS_START = "2024-11-01"
ANALYSIS_END   = "2025-03-31"

# %% [markdown]
# ## Step 1 — Load ERA5 Z500

# %%
era5 = xr.open_dataset(paths.era5_z500_nc(ANALYSIS_START, ANALYSIS_END))
z0500 = era5["z0500"]   # (time, lat, lon), geopotential height in metres, lat N→S

print(f"ERA5 z0500  shape : {z0500.shape}")
print(f"             time : {str(z0500.time.values[0])[:16]} → {str(z0500.time.values[-1])[:16]}")
print(f"              lat : {float(z0500.latitude[0]):.1f} → {float(z0500.latitude[-1]):.1f}")
print(f"              lon : {float(z0500.longitude[0]):.1f} → {float(z0500.longitude[-1]):.1f}")

# %% [markdown]
# ## Step 2 — Subtract year-round climatological mean

# %%
# Climatology is in geopotential (m²/s²); divide by g to get geopotential height (m)
G = 9.80665
with nc.Dataset(paths.grams_clim_z500) as ds:
    clim_lat = ds.variables["latitude"][:]   # -90 → 90
    clim_lon = ds.variables["longitude"][:]  # -180 → 179.5
    clim_z   = ds.variables["Z@500"][:] / G  # (361, 720), metres

# Crop to EOF domain matching ERA5 download (lat 90→30°N, lon -80→40°E)
lat_mask = (clim_lat >= 30.0) & (clim_lat <= 90.0)
lon_mask = (clim_lon >= -80.0) & (clim_lon <= 40.0)
clim_crop = clim_z[np.ix_(lat_mask, lon_mask)]        # (121, 241), S→N
clim_crop = clim_crop[::-1, :]                         # flip to N→S to match ERA5

print(f"Climatology shape after crop+flip: {clim_crop.shape}  (lat N→S, lon W→E)")
print(f"Climatology range: {clim_crop.min():.0f} – {clim_crop.max():.0f} m")

# Subtract (broadcast over time)
z_anom = z0500.values - clim_crop[np.newaxis, :, :]   # (time, lat, lon)
print(f"Anomaly range: {z_anom.min():.0f} – {z_anom.max():.0f} m")

# %% [markdown]
# ## Step 3 — Apply 161-point Lanczos low-pass filter (cutoff 240 h)

# %%
def _lanczos_lp_weights(n: int, cutoff_ts: float) -> np.ndarray:
    """Lanczos LP filter weights.

    n          : number of weights (must be odd)
    cutoff_ts  : cutoff period in timesteps (240 h / 3 h = 80)
    """
    m = (n - 1) // 2
    k = np.arange(-m, m + 1, dtype=float)
    fc = 1.0 / cutoff_ts
    # Ideal sinc LP weights (k=0 handled separately to avoid 0/0)
    k_nz = np.where(k == 0, 1.0, k)
    w     = np.sin(2 * np.pi * fc * k_nz) / (np.pi * k_nz)
    w[m]  = 2 * fc
    # Lanczos (sinc) window
    sigma     = np.sin(np.pi * k_nz / m) / (np.pi * k_nz / m)
    sigma[m]  = 1.0
    w *= sigma
    w /= w.sum()
    return w


N_WEIGHTS   = 161
CUTOFF_H    = 240          # hours
DT_H        = 3            # hours per timestep
CUTOFF_TS   = CUTOFF_H / DT_H   # = 80 timesteps

weights = _lanczos_lp_weights(N_WEIGHTS, CUTOFF_TS)
print(f"Filter: {N_WEIGHTS} weights, cutoff {CUTOFF_H} h ({CUTOFF_TS:.0f} timesteps)")
print(f"Guard zone each side: {(N_WEIGHTS - 1) // 2} timesteps = {(N_WEIGHTS - 1) // 2 * DT_H / 24:.0f} days")

# Apply along time axis; boundary timesteps within the guard zone are invalid
z_lp = convolve1d(z_anom, weights, axis=0, mode="constant", cval=0.0)
print(f"LP-filtered shape: {z_lp.shape}")

# %% [markdown]
# ## Step 4 — Crop to analysis period

# %%
times      = pd.DatetimeIndex(z0500.time.values)
t_start    = pd.Timestamp(ANALYSIS_START)
t_end      = pd.Timestamp(ANALYSIS_END) + pd.Timedelta("21h")   # include 21 UTC
ana_mask   = (times >= t_start) & (times <= t_end)

z_lp_ana   = z_lp[ana_mask]          # (n_ana, 121, 241)
times_ana  = times[ana_mask]

print(f"Analysis period: {times_ana[0]} → {times_ana[-1]}  ({len(times_ana)} timesteps)")

# %% [markdown]
# ## Step 5 — Normalise by seasonal weight `normwgt`

# %%
with nc.Dataset(paths.grams_eofs) as ds:
    normwgt  = ds.variables["normwgt"][:]   # (1464,) 6-hourly, one leap year
    nrmtimes = ds.variables["time"][:]      # hours since 1980-01-01 00:00


def _normwgt_index(ts: pd.Timestamp, nrmtimes: np.ndarray) -> int:
    """Return normwgt array index for a given timestamp."""
    tdiff = int((ts - pd.Timestamp(f"{ts.year}-01-01")).total_seconds() / 3600)
    tdiff -= tdiff % 6   # snap to 6-h grid (03→00, 09→06, 15→12, 21→18)
    if tdiff >= nrmtimes[236] and not calendar.isleap(ts.year):
        return int(np.where(nrmtimes == tdiff + 24)[0][0])
    return int(np.where(nrmtimes == tdiff)[0][0])


nwgt_vals = np.array([normwgt[_normwgt_index(t, nrmtimes)] for t in times_ana])
print(f"normwgt  min={nwgt_vals.min():.4f}  max={nwgt_vals.max():.4f}  "
      f"mean={nwgt_vals.mean():.4f}")

z_norm = z_lp_ana / nwgt_vals[:, np.newaxis, np.newaxis]   # (n_ana, 121, 241)

# %% [markdown]
# ## Step 6 — Project onto regime patterns

# %%
with nc.Dataset(paths.grams_z500_patterns) as ds:
    patterns  = ds.variables["Z0500_mean"][:]   # (7, 121, 241), S→N order
    pat_names = ds.ClassNames.split()           # ['AT','ZO','ScTr','AR','EuBL','ScBL','GL']
    lat_eof   = ds.variables["latitude"][:]     # 30.0 … 90.0 (S→N)
    lon_eof   = ds.variables["longitude"][:]    # -80.0 … 40.0

# ERA5 lat is N→S; flip z_norm so it is S→N to match patterns
z_norm_sn = z_norm[:, ::-1, :]   # (n_ana, 121, 241), now S→N

# Cosine-latitude weights (2-D, normalised to sum=1), computed on the pattern grid
d2r   = np.pi / 180.0
cos2d = np.tile(np.cos(lat_eof * d2r)[:, np.newaxis], (1, len(lon_eof)))
cos2d /= cos2d.sum()

# Project: (n_ana,) per regime
proj = np.zeros((len(times_ana), 7))
for w, name in enumerate(pat_names):
    i_wr = WR_NAMES.index(name)
    proj[:, i_wr] = np.einsum("ij,tij->t", cos2d * patterns[w], z_norm_sn)

print(f"Projection range: {proj.min():.3f} – {proj.max():.3f}")

# %% [markdown]
# ## Step 7 — Standardise → IWR

# %%
with open(paths.grams_std_params) as f:
    f.readline()
    std_names = f.readline().split()
    std_mean  = np.array(f.readline().split()[1:], dtype=float)
    std_std   = np.array(f.readline().split()[1:], dtype=float)

iwr = np.zeros_like(proj)
for w, name in enumerate(WR_NAMES):
    i_std = std_names.index(name)
    iwr[:, w] = (proj[:, w] - std_mean[i_std]) / std_std[i_std]

iwr_df = pd.DataFrame(iwr, index=times_ana, columns=WR_NAMES)
print(iwr_df.describe().round(3))

# %% [markdown]
# ## Step 8 — Compare with pre-computed Grams IWR

# %%
ref = pd.read_csv(paths.wri_csv, parse_dates=["datetime"], index_col="datetime")
ref_ana = ref.loc[t_start:t_end, WR_NAMES]

# Align on common timestamps
common = iwr_df.index.intersection(ref_ana.index)
diff   = iwr_df.loc[common] - ref_ana.loc[common]

print(f"Common timesteps: {len(common)}")
print(f"\nMean absolute error per regime:")
print(diff.abs().mean().round(4).to_string())
print(f"\nOverall MAE : {diff.abs().values.mean():.4f}")
print(f"Overall RMSE: {np.sqrt((diff.values ** 2).mean()):.4f}")
print(f"Correlation :")
for reg in WR_NAMES:
    r = np.corrcoef(iwr_df.loc[common, reg], ref_ana.loc[common, reg])[0, 1]
    print(f"  {reg:5s}: r = {r:.4f}")

# %% [markdown]
# ## Step 9 — Plot: ERA5-computed vs pre-computed IWR

# %%
fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)

for ax, reg in zip(axes, WR_NAMES):
    ax.plot(common, ref_ana.loc[common, reg],
            color="0.5", lw=1.0, label="Grams pre-computed")
    ax.plot(common, iwr_df.loc[common, reg],
            color="steelblue", lw=1.2, alpha=0.85, label="ERA5 recomputed")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(1, color="k", lw=0.5, ls="--")
    ax.set_ylabel(reg, fontsize=10)
    ax.set_ylim(-3.5, 3.5)
    r = np.corrcoef(iwr_df.loc[common, reg], ref_ana.loc[common, reg])[0, 1]
    mae = diff.loc[common, reg].abs().mean()
    ax.text(0.01, 0.92, f"r={r:.3f}  MAE={mae:.3f}",
            transform=ax.transAxes, fontsize=9, va="top")

axes[0].legend(loc="upper right", fontsize=9)
axes[0].set_title(
    f"IWR: ERA5-recomputed vs Grams pre-computed  "
    f"({ANALYSIS_START} – {ANALYSIS_END})",
    fontsize=11,
)
axes[-1].set_xlabel("Date")

fig.tight_layout()
fig.savefig(paths.images_path / "06_era5_iwr_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/06_era5_iwr_comparison.png
# :name: fig-06-era5-iwr-comparison
# IWR time series computed from raw ERA5 Z500 (blue) versus the pre-computed
# Grams reference (grey) for November 2024 – March 2025.  Differences arise
# primarily from using the year-round climatological mean instead of a
# grid-point-wise daily climatology for the anomaly computation.
# ```
