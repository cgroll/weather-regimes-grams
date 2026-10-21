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
# # ERA5 WR Projections with a Seasonal Climatology
#
# Script 06 recomputes the IWR from raw ERA5 Z500 but forms anomalies with the
# **year-round** Grams mean — the only climatology shipped in the Zenodo
# package.  The recomputed IWR correlate well with the Grams reference
# (r ≈ 0.94–0.997) but show large offsets (MAE ≈ 0.57).
#
# Hypothesis: the offsets come from the annual cycle that remains in the
# anomaly.  This notebook tests that by repeating the pipeline with a
# seasonally varying (day-of-year × hour-of-day) climatology from
# WeatherBench2 (ERA5, 1990–2019, 6-hourly, see script 24).
#
# ## Variants compared
#
# | Variant | Climatology subtracted |
# |---------|------------------------|
# | **A — year-round** | Grams `CLIM_Z@500_year_1979-2019.nc` (as in script 06) |
# | **B — WB2 seasonal** | WB2 day-of-year × hour climatology, 1990–2019 |
# | **C — WB2 seasonal, Grams mean** | B shifted so that its annual mean equals the Grams year-round mean |
#
# Variant C removes the difference in reference period (1990–2019 vs
# 1979–2019) from the annual mean while keeping the WB2 seasonal shape.
#
# All other steps (Lanczos filter, `normwgt`, projection, standardisation) are
# identical to script 06.

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
G = 9.80665

# %% [markdown]
# ## Step 1 — Load ERA5 Z500 and the two climatologies

# %%
era5  = xr.open_dataset(paths.era5_z500_nc(ANALYSIS_START, ANALYSIS_END))
z0500 = era5["z0500"]   # (time, lat, lon), metres, lat N→S
times = pd.DatetimeIndex(z0500.time.values)
print(f"ERA5 z0500 shape : {z0500.shape}  ({times[0]} → {times[-1]})")

# Grams year-round mean (geopotential, m²/s²) → metres, cropped, N→S
with nc.Dataset(paths.grams_clim_z500) as ds:
    clim_lat = ds.variables["latitude"][:]   # -90 → 90
    clim_lon = ds.variables["longitude"][:]  # -180 → 179.5
    clim_z   = ds.variables["Z@500"][:] / G
lat_mask   = (clim_lat >= 30.0) & (clim_lat <= 90.0)
lon_mask   = (clim_lon >= -80.0) & (clim_lon <= 40.0)
clim_year  = np.asarray(clim_z[np.ix_(lat_mask, lon_mask)])[::-1, :]   # (121, 241), N→S

# WB2 seasonal climatology (hour × dayofyear × lat × lon), S→N → flip to N→S
wb = xr.open_zarr(paths.wb_z500_climatology_6h)["z500"].sortby("latitude", ascending=False)
assert np.allclose(wb.latitude.values, z0500.latitude.values)
assert np.allclose(wb.longitude.values, z0500.longitude.values)
wb_hours = list(wb.hour.values)
clim_wb  = wb.values                                               # (4, 366, 121, 241)
print(f"WB2 climatology  : {clim_wb.shape}  hours={wb_hours}")

# %% [markdown]
# ### Seasonal climatology at the 3-hourly ERA5 timesteps
#
# WB2 provides 00/06/12/18 UTC.  For 03/09/15/21 UTC the climatology is the
# mean of the two neighbouring 6-hourly fields (the Z500 diurnal cycle is
# small, so linear interpolation is sufficient).  The day-of-year index
# follows `pandas.dayofyear`, the same convention WB2 uses to build the
# climatology.

# %%
def _wb_field(ts: pd.Timestamp) -> np.ndarray:
    return clim_wb[wb_hours.index(ts.hour), ts.dayofyear - 1]


def wb_clim_at(ts: pd.Timestamp) -> np.ndarray:
    if ts.hour % 6 == 0:
        return _wb_field(ts)
    return 0.5 * (_wb_field(ts - pd.Timedelta("3h")) + _wb_field(ts + pd.Timedelta("3h")))


clim_seasonal = np.stack([wb_clim_at(t) for t in times])   # (time, 121, 241)

# Annual mean of WB2 (over all days and hours, weighting doy 366 by 1/4)
doy_w = np.ones(366)
doy_w[365] = 0.25
wb_annual_mean = np.einsum("d,hdij->ij", doy_w, clim_wb) / (4 * doy_w.sum())
offset = clim_year - wb_annual_mean   # Grams 1979–2019 minus WB2 1990–2019

print(f"Annual mean offset Grams − WB2 : mean={offset.mean():.1f} m  "
      f"min={offset.min():.1f}  max={offset.max():.1f}")
print(f"Seasonal anomaly of WB2 clim in analysis window (clim − annual mean): "
      f"{(clim_seasonal - wb_annual_mean).min():.0f} … "
      f"{(clim_seasonal - wb_annual_mean).max():.0f} m")

# %% [markdown]
# ## Step 2 — Shared projection pipeline
#
# Lanczos LP filter → crop → `normwgt` → projection → standardisation,
# exactly as in script 06.

# %%
def _lanczos_lp_weights(n: int, cutoff_ts: float) -> np.ndarray:
    """Lanczos LP filter weights (n odd, cutoff in timesteps)."""
    m = (n - 1) // 2
    k = np.arange(-m, m + 1, dtype=float)
    fc = 1.0 / cutoff_ts
    k_nz = np.where(k == 0, 1.0, k)
    w     = np.sin(2 * np.pi * fc * k_nz) / (np.pi * k_nz)
    w[m]  = 2 * fc
    sigma     = np.sin(np.pi * k_nz / m) / (np.pi * k_nz / m)
    sigma[m]  = 1.0
    w *= sigma
    w /= w.sum()
    return w


def _normwgt_index(ts: pd.Timestamp, nrmtimes: np.ndarray) -> int:
    """Return normwgt array index for a given timestamp."""
    tdiff = int((ts - pd.Timestamp(f"{ts.year}-01-01")).total_seconds() / 3600)
    tdiff -= tdiff % 6
    if tdiff >= nrmtimes[236] and not calendar.isleap(ts.year):
        return int(np.where(nrmtimes == tdiff + 24)[0][0])
    return int(np.where(nrmtimes == tdiff)[0][0])


weights = _lanczos_lp_weights(161, 240 / 3)

t_start  = pd.Timestamp(ANALYSIS_START)
t_end    = pd.Timestamp(ANALYSIS_END) + pd.Timedelta("21h")
ana_mask = (times >= t_start) & (times <= t_end)
times_ana = times[ana_mask]

with nc.Dataset(paths.grams_eofs) as ds:
    normwgt  = ds.variables["normwgt"][:]
    nrmtimes = ds.variables["time"][:]
nwgt_vals = np.array([normwgt[_normwgt_index(t, nrmtimes)] for t in times_ana])

with nc.Dataset(paths.grams_z500_patterns) as ds:
    patterns  = ds.variables["Z0500_mean"][:]   # (7, 121, 241), S→N
    pat_names = ds.ClassNames.split()
    lat_eof   = ds.variables["latitude"][:]
    lon_eof   = ds.variables["longitude"][:]

cos2d = np.tile(np.cos(np.deg2rad(lat_eof))[:, np.newaxis], (1, len(lon_eof)))
cos2d /= cos2d.sum()

with open(paths.grams_std_params) as f:
    f.readline()
    std_names = f.readline().split()
    std_mean  = np.array(f.readline().split()[1:], dtype=float)
    std_std   = np.array(f.readline().split()[1:], dtype=float)


def compute_iwr(z_anom: np.ndarray) -> pd.DataFrame:
    """Full-period anomaly (time, lat N→S, lon) → IWR DataFrame on times_ana."""
    z_lp   = convolve1d(z_anom, weights, axis=0, mode="constant", cval=0.0)
    z_norm = z_lp[ana_mask] / nwgt_vals[:, np.newaxis, np.newaxis]
    z_sn   = z_norm[:, ::-1, :]
    iwr = np.zeros((len(times_ana), 7))
    for w, name in enumerate(pat_names):
        i_wr  = WR_NAMES.index(name)
        i_std = std_names.index(name)
        proj  = np.einsum("ij,tij->t", cos2d * patterns[w], z_sn)
        iwr[:, i_wr] = (proj - std_mean[i_std]) / std_std[i_std]
    return pd.DataFrame(iwr, index=times_ana, columns=WR_NAMES)


# %% [markdown]
# ## Step 3 — Compute IWR for all three variants

# %%
z = z0500.values
variants = {
    "A year-round":            compute_iwr(z - clim_year[np.newaxis]),
    "B WB2 seasonal":          compute_iwr(z - clim_seasonal),
    "C WB2 seasonal, Grams mean": compute_iwr(z - clim_seasonal - offset[np.newaxis]),
}

ref = pd.read_csv(paths.wri_csv, parse_dates=["datetime"], index_col="datetime")
ref_ana = ref.loc[t_start:t_end, WR_NAMES]
common  = times_ana.intersection(ref_ana.index)
print(f"Common timesteps: {len(common)}")

# %% [markdown]
# ## Step 4 — Compare with the Grams reference

# %%
rows = []
for vname, iwr_df in variants.items():
    diff = iwr_df.loc[common] - ref_ana.loc[common]
    for reg in WR_NAMES:
        rows.append({
            "variant": vname,
            "regime":  reg,
            "bias":    diff[reg].mean(),
            "MAE":     diff[reg].abs().mean(),
            "r":       np.corrcoef(iwr_df.loc[common, reg], ref_ana.loc[common, reg])[0, 1],
        })
metrics = pd.DataFrame(rows)

for stat in ["bias", "MAE", "r"]:
    print(f"\n{stat} per regime")
    print(metrics.pivot(index="variant", columns="regime", values=stat)[WR_NAMES]
          .round(3).to_string())

summary = metrics.groupby("variant").agg(
    MAE=("MAE", "mean"),
    abs_bias=("bias", lambda s: s.abs().mean()),
    r_min=("r", "min"),
)
summary["RMSE"] = [
    np.sqrt(((variants[v].loc[common] - ref_ana.loc[common]) ** 2).values.mean())
    for v in summary.index
]
print("\nSummary")
print(summary.round(4).to_string())

# %% [markdown]
# ## Step 5 — Plot: time series for all variants

# %%
colors = {
    "A year-round":               "tab:red",
    "B WB2 seasonal":             "tab:orange",
    "C WB2 seasonal, Grams mean": "steelblue",
}

fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)
for ax, reg in zip(axes, WR_NAMES):
    ax.plot(common, ref_ana.loc[common, reg], color="0.3", lw=1.6, label="Grams pre-computed")
    for vname, iwr_df in variants.items():
        ax.plot(common, iwr_df.loc[common, reg], color=colors[vname], lw=1.0,
                alpha=0.85, label=vname)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(1, color="k", lw=0.5, ls="--")
    ax.set_ylabel(reg, fontsize=10)
    ax.set_ylim(-3.5, 3.5)
    m = metrics[metrics.regime == reg].set_index("variant")["MAE"]
    ax.text(0.01, 0.92, "MAE  " + "   ".join(f"{v[0]}={m[v]:.3f}" for v in variants),
            transform=ax.transAxes, fontsize=9, va="top")

axes[0].legend(loc="upper right", fontsize=8, ncol=2)
axes[0].set_title(
    f"IWR recomputed from ERA5 with different climatologies vs Grams  "
    f"({ANALYSIS_START} – {ANALYSIS_END})",
    fontsize=11,
)
axes[-1].set_xlabel("Date")
fig.tight_layout()
fig.savefig(paths.images_path / "25_era5_iwr_clim_variants.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/25_era5_iwr_clim_variants.png
# :name: fig-25-era5-iwr-clim-variants
# IWR recomputed from ERA5 Z500 with three climatologies, compared with the
# Grams reference (dark grey).  A: Grams year-round mean (as in script 06).
# B: WeatherBench2 day-of-year × hour climatology (1990–2019).  C: as B, but
# shifted to the Grams 1979–2019 annual mean.
# ```

# %% [markdown]
# ## Step 6 — Error per regime and variant

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, stat in zip(axes, ["bias", "MAE"]):
    pv = metrics.pivot(index="regime", columns="variant", values=stat).loc[WR_NAMES]
    x  = np.arange(len(WR_NAMES))
    bw = 0.8 / len(variants)
    for i, vname in enumerate(variants):
        ax.bar(x + (i - 1) * bw, pv[vname], bw, color=colors[vname], label=vname)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(WR_NAMES)
    ax.set_title(f"{stat} (recomputed − Grams)")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(paths.images_path / "25_era5_iwr_clim_errors.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/25_era5_iwr_clim_errors.png
# :name: fig-25-era5-iwr-clim-errors
# Mean bias and mean absolute error of the recomputed IWR per regime for the
# three climatology variants.
# ```
