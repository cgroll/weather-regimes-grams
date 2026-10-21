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
# # Full IWR Time Series Recomputed from ERA5 (1960–2021)
#
# Scripts 25 and 27 show that the Grams IWR can be closely reproduced from
# raw ERA5 once a seasonal (WeatherBench2) climatology is used — but only
# for one winter (2024/25) and one timestep (2025-06-01).  This notebook
# recomputes the **whole** IWR record for all 7 regimes and compares it with
# the published Grams series over six decades.
#
# ## Data actually used (no new download)
#
# | | Grams | here |
# |---|---|---|
# | Z500 | ERA5, 3-hourly, 0.5° | ERA5 via WeatherBench2, **daily 12 UTC, 1.5°** (scripts 16–17) |
# | Climatology | own, 1979–2019 | WB2 day-of-year climatology at 12 UTC, 1990–2019, 1.5° (script 15) |
# | LP filter | Lanczos, 161 pts at Δt = 3 h, cutoff 10 d | Lanczos, 21 pts at Δt = 24 h (same ±10 d span), cutoff 10 d |
# | Projection | 0.5° EOF domain | anomaly bilinearly interpolated to the same 0.5° grid, then identical projection |
#
# The exact 3-hourly / 0.5° data for 1950–2025 would require ≈ 17 TB of ARCO
# transfer, so this is a deliberately cheap approximation.  Differences can
# come from the climatology (period, smoothing), the coarser grid, and the
# daily sampling before filtering.
#
# Comparison period: 1960-01-11 → 2021-12-21 (10-day filter guard at both
# ends), Grams values at 12 UTC.

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

# %% [markdown]
# ## Step 1 — Daily anomalies on the 1.5° grid

# %%
z = xr.open_zarr(paths.era5_z500_daily_zarr)["z500"].transpose("time", "latitude", "longitude").load()
clim = xr.open_zarr(paths.wb_z500_climatology)["z500"].transpose("dayofyear", "latitude", "longitude").load()
assert np.allclose(z.latitude, clim.latitude) and np.allclose(z.longitude, clim.longitude)

times = pd.DatetimeIndex(z.time.values)
assert (times.hour == 12).all()
doy = times.dayofyear.values
anom = z.values - clim.values[doy - 1]     # (time, 41, 81), lat S→N
print(f"ERA5 daily: {z.shape}  {times[0]:%Y-%m-%d} → {times[-1]:%Y-%m-%d}")
print(f"Anomaly range: {anom.min():.0f} … {anom.max():.0f} m")

# %% [markdown]
# ## Step 2 — Lanczos LP filter at daily resolution
#
# Same cutoff (240 h = 10 timesteps of 24 h) and same ±10-day span as the
# Grams 161-point filter at 3 h.

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


weights = _lanczos_lp_weights(21, 240 / 24)
m = (len(weights) - 1) // 2
assert (np.diff(times) == np.timedelta64(1, "D")).all(), "daily series must be gap-free"

anom_lp = convolve1d(anom, weights, axis=0, mode="constant", cval=0.0)[m:-m]
times_lp = times[m:-m]
print(f"LP-filtered: {anom_lp.shape}  {times_lp[0]:%Y-%m-%d} → {times_lp[-1]:%Y-%m-%d}")

# %% [markdown]
# ## Step 3 — Interpolate to the 0.5° pattern grid and project

# %%
with nc.Dataset(paths.grams_z500_patterns) as ds:
    patterns  = np.asarray(ds.variables["Z0500_mean"][:])   # (7, 121, 241), S→N
    pat_names = ds.ClassNames.split()
    lat_eof   = np.asarray(ds.variables["latitude"][:])
    lon_eof   = np.asarray(ds.variables["longitude"][:])

cos2d = np.tile(np.cos(np.deg2rad(lat_eof))[:, np.newaxis], (1, len(lon_eof)))
cos2d /= cos2d.sum()

with nc.Dataset(paths.grams_eofs) as ds:
    normwgt  = ds.variables["normwgt"][:]
    nrmtimes = ds.variables["time"][:]


def _normwgt_index(ts: pd.Timestamp) -> int:
    tdiff = int((ts - pd.Timestamp(f"{ts.year}-01-01")).total_seconds() / 3600)
    tdiff -= tdiff % 6
    if tdiff >= nrmtimes[236] and not calendar.isleap(ts.year):
        return int(np.where(nrmtimes == tdiff + 24)[0][0])
    return int(np.where(nrmtimes == tdiff)[0][0])


nwgt_vals = np.array([normwgt[_normwgt_index(t)] for t in times_lp])

with open(paths.grams_std_params) as f:
    f.readline()
    std_names = f.readline().split()
    std_mean  = np.array(f.readline().split()[1:], dtype=float)
    std_std   = np.array(f.readline().split()[1:], dtype=float)

# Projecting the bilinearly interpolated field equals projecting the 1.5°
# field onto "interpolation-adjoint" patterns, so we interpolate the 7
# weighted patterns to the 1.5° grid once instead of interpolating 22 000
# fields.  The 0.5° grid lies inside the 1.5° grid except for the −80° column
# (1.5° grid starts at −79.5°), which is filled by nearest-neighbour.
lat15, lon15 = z.latitude.values, z.longitude.values


def _interp_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Linear interpolation matrix (len(dst) × len(src)), clamped at the edges."""
    W = np.zeros((len(dst), len(src)))
    for i, x in enumerate(dst):
        x = np.clip(x, src[0], src[-1])
        j = min(np.searchsorted(src, x, side="right") - 1, len(src) - 2)
        t = (x - src[j]) / (src[j + 1] - src[j])
        W[i, j], W[i, j + 1] = 1 - t, t
    return W


W_lat = _interp_matrix(lat15, lat_eof)   # (121, 41)
W_lon = _interp_matrix(lon15, lon_eof)   # (241, 81)

iwr = np.zeros((len(times_lp), 7))
for w, name in enumerate(pat_names):
    adj = W_lat.T @ (cos2d * patterns[w]) @ W_lon        # (41, 81)
    proj = np.einsum("ij,tij->t", adj, anom_lp) / nwgt_vals
    i_std = std_names.index(name)
    iwr[:, WR_NAMES.index(name)] = (proj - std_mean[i_std]) / std_std[i_std]

recomp = pd.DataFrame(iwr, index=times_lp, columns=WR_NAMES)

ref = pd.read_csv(paths.wri_csv, parse_dates=["datetime"], index_col="datetime")[WR_NAMES]
common = recomp.index.intersection(ref.index)
recomp, grams = recomp.loc[common], ref.loc[common]
diff = recomp - grams
print(f"Common daily timesteps (12 UTC): {len(common):,}  ({common[0]:%Y-%m-%d} → {common[-1]:%Y-%m-%d})")

# %% [markdown]
# ## Step 4 — Metrics per regime

# %%
metrics = pd.DataFrame({
    "bias":  diff.mean(),
    "MAE":   diff.abs().mean(),
    "RMSE":  np.sqrt((diff ** 2).mean()),
    "p99 |Δ|": diff.abs().quantile(0.99),
    "max |Δ|": diff.abs().max(),
    "r":     recomp.corrwith(grams),
    "std Grams": grams.std(),
}).T[WR_NAMES]
metrics["all"] = [diff.values.mean(), np.abs(diff.values).mean(), np.sqrt((diff.values ** 2).mean()),
                  np.quantile(np.abs(diff.values), 0.99), np.abs(diff.values).max(),
                  np.corrcoef(recomp.values.ravel(), grams.values.ravel())[0, 1], grams.values.std()]
print(metrics.round(3).to_string())

# Regime classification agreement
argmax_match = (recomp.idxmax(axis=1) == grams.idxmax(axis=1)).mean()
above1_match = ((recomp >= 1) == (grams >= 1)).mean()
print(f"\nMax-IWR regime identical : {100 * argmax_match:.2f} % of days")
print("IWR ≥ 1 classification identical per regime (%):")
print((100 * above1_match).round(2).to_string())

# %% [markdown]
# ## Step 5 — Over-time comparison per regime
#
# Six decades of daily values are too dense to read directly, so each panel
# shows the **monthly means** of both series (top line pair) and, below, the
# daily difference with its 365-day rolling mean.  Slow drifts in the rolling
# mean point to climatology issues (reference period, trends); spikes point
# to individual events.

# %%
grams_m  = grams.resample("MS").mean()
recomp_m = recomp.resample("MS").mean()
roll     = diff.rolling(365, center=True, min_periods=200).mean()

fig, axes = plt.subplots(14, 1, figsize=(15, 30), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1] * 7})
for k, reg in enumerate(WR_NAMES):
    ax, axd = axes[2 * k], axes[2 * k + 1]
    ax.plot(grams_m.index, grams_m[reg], color="0.3", lw=1.0, label="Grams (monthly mean)")
    ax.plot(recomp_m.index, recomp_m[reg], color="steelblue", lw=0.9, alpha=0.85,
            label="recomputed (monthly mean)")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel(f"{reg}\nIWR", fontsize=9)
    ax.text(0.005, 0.93,
            f"r={metrics.loc['r', reg]:.3f}  MAE={metrics.loc['MAE', reg]:.3f}  "
            f"bias={metrics.loc['bias', reg]:+.3f}",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    axd.plot(diff.index, diff[reg], color="0.75", lw=0.3)
    axd.plot(roll.index, roll[reg], color="tab:red", lw=1.2)
    axd.axhline(0, color="k", lw=0.4)
    axd.set_ylim(-1, 1)
    axd.set_ylabel("Δ", fontsize=9)
axes[0].legend(loc="upper right", fontsize=8, ncol=2)
axes[0].set_title("IWR: recomputed (ERA5 daily 1.5° + WB2 climatology) vs Grams, 1960–2021\n"
                  "Δ panels: daily recomputed − Grams (grey) and 365-day rolling mean (red)",
                  fontsize=11)
fig.tight_layout()
fig.savefig(paths.images_path / "28_iwr_recomputed_timeseries.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/28_iwr_recomputed_timeseries.png
# :name: fig-28-iwr-recomputed-timeseries
# Monthly mean IWR of the Grams series (grey) and the recomputed series (blue)
# for each regime, with the daily difference and its 365-day rolling mean
# below.
# ```

# %% [markdown]
# ### Zoom: one year at daily resolution
#
# The year with the median overall MAE is shown, so the panel is
# representative rather than a best or worst case.

# %%
mae_year = diff.abs().mean(axis=1).groupby(diff.index.year).mean()
year = int(mae_year.sort_values().index[len(mae_year) // 2])
sl = slice(f"{year}-01-01", f"{year}-12-31")

fig, axes = plt.subplots(7, 1, figsize=(14, 16), sharex=True)
for ax, reg in zip(axes, WR_NAMES):
    ax.plot(grams.loc[sl].index, grams.loc[sl, reg], color="0.3", lw=1.4, label="Grams")
    ax.plot(recomp.loc[sl].index, recomp.loc[sl, reg], color="steelblue", lw=1.0, label="recomputed")
    ax.axhline(0, color="k", lw=0.4)
    ax.axhline(1, color="k", lw=0.4, ls="--")
    ax.set_ylabel(reg)
    ax.set_ylim(-3.5, 3.5)
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_title(f"IWR {year} (year with median MAE {mae_year[year]:.3f}), daily 12 UTC", fontsize=11)
fig.tight_layout()
fig.savefig(paths.images_path / "28_iwr_recomputed_zoom.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/28_iwr_recomputed_zoom.png
# :name: fig-28-iwr-recomputed-zoom
# Daily IWR for a representative year: Grams (grey) and recomputed (blue).
# ```

# %% [markdown]
# ## Step 6 — Scatter per regime

# %%
fig, axes = plt.subplots(2, 4, figsize=(17, 8.8))
for ax, reg in zip(axes.ravel(), WR_NAMES):
    ax.hexbin(grams[reg], recomp[reg], gridsize=80, bins="log", cmap="Blues", mincnt=1,
              extent=(-4, 4, -4, 4))
    ax.plot([-4, 4], [-4, 4], color="0.2", lw=0.8, ls="--")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
    ax.set_title(reg, fontsize=11)
    ax.set_xlabel("Grams IWR", fontsize=9)
    ax.set_ylabel("recomputed IWR", fontsize=9)
    ax.text(0.03, 0.97, f"r={metrics.loc['r', reg]:.4f}\nRMSE={metrics.loc['RMSE', reg]:.3f}",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="0.8"))
    ax.grid(color="0.92", lw=0.5)

# Last panel: MAE per decade and regime
ax = axes.ravel()[-1]
dec = diff.abs().groupby((diff.index.year // 10) * 10).mean()[WR_NAMES]
im = ax.imshow(dec.values, cmap="Reds", aspect="auto", vmin=0)
ax.set_xticks(range(7)); ax.set_xticklabels(WR_NAMES, fontsize=8)
ax.set_yticks(range(len(dec))); ax.set_yticklabels([f"{d}s" for d in dec.index], fontsize=8)
for (i, j), v in np.ndenumerate(dec.values):
    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
            color="white" if v > 0.6 * dec.values.max() else "0.15")
ax.set_title("MAE per decade", fontsize=11)
fig.tight_layout()
fig.savefig(paths.images_path / "28_iwr_recomputed_scatter.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/28_iwr_recomputed_scatter.png
# :name: fig-28-iwr-recomputed-scatter
# Recomputed vs Grams IWR for each regime (all days 1960–2021, 12 UTC); the
# dashed line marks perfect agreement.  Bottom right: mean absolute error per
# decade and regime.
# ```

# %% [markdown]
# ## Step 7 — Largest deviations
#
# The 15 days with the largest absolute difference over all regimes.

# %%
long = diff.stack().rename("delta").reset_index()
long.columns = ["date", "regime", "delta"]
long["grams"]  = [grams.at[d, r] for d, r in zip(long.date, long.regime)]
long["recomp"] = [recomp.at[d, r] for d, r in zip(long.date, long.regime)]
top = long.reindex(long.delta.abs().sort_values(ascending=False).index).head(15)
print(top.round(3).to_string(index=False))

# %% [markdown]
# ## Step 8 — Error by calendar day
#
# The Δ panels above show spikes that recur every year at the same season,
# with no drift across decades.  Averaging Δ over all years per day of year
# isolates this seasonal error component.

# %%
by_doy  = diff.groupby(diff.index.dayofyear).mean().loc[1:365]
mae_doy = diff.abs().groupby(diff.index.dayofyear).mean().loc[1:365]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for reg in WR_NAMES:
    axes[0].plot(by_doy.index, by_doy[reg], lw=1.2, label=reg)
    axes[1].plot(mae_doy.index, mae_doy[reg], lw=1.2, label=reg)
axes[0].axhline(0, color="k", lw=0.5)
axes[0].set_ylabel("mean Δ (recomputed − Grams)")
axes[1].set_ylabel("MAE")
axes[1].set_xlabel("day of year")
month_starts = pd.date_range("2001-01-01", periods=12, freq="MS")
axes[1].set_xticks(month_starts.dayofyear)
axes[1].set_xticklabels(month_starts.strftime("%b"))
axes[0].legend(ncol=7, fontsize=8, loc="upper left")
axes[0].set_title("Seasonal structure of the IWR error (mean over 1960–2021 per calendar day)", fontsize=11)
for ax in axes:
    ax.grid(color="0.92", lw=0.5)
fig.tight_layout()
fig.savefig(paths.images_path / "28_iwr_recomputed_doy.png", dpi=130, bbox_inches="tight")
plt.show()

print("Day of year with the largest mean |Δ| per regime:")
print(pd.DataFrame({"doy": by_doy.abs().idxmax(),
                    "date": [pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(d) - 1)
                             for d in by_doy.abs().idxmax()],
                    "mean Δ": [by_doy.at[d, r] for r, d in by_doy.abs().idxmax().items()]})
      .assign(date=lambda d: d.date.dt.strftime("%d %b")).round(3).to_string())

# %% [markdown]
# ```{figure} ../../output/images/28_iwr_recomputed_doy.png
# :name: fig-28-iwr-recomputed-doy
# Mean difference and mean absolute difference per calendar day, averaged
# over all years 1960–2021.
# ```

# %% [markdown]
# ## Step 9 — Systematic climatology bias: estimate and remove
#
# The anomaly → LP filter → projection chain is linear, so any difference
# between the WB2 and the Grams climatology enters the IWR as a fixed offset
# per calendar day and regime.  The climatology field itself cannot be
# recovered from 7 IWR values per timestep, but the offset can: it is the
# mean Δ per calendar day over all years (the top panel of the previous
# figure).
#
# Here the offset is estimated **in-sample** on the full 1960–2021 record and
# subtracted.  What remains is a proxy for the error that is *not* due to the
# climatology (coarser grid, daily sampling, filter); the difference between
# the two is the systematic share of the climatology.  Day 366 uses the
# offset of day 365.  An out-of-sample check (fit 1960–1990, test 1991–2021)
# is printed alongside.

# %%
doy_all  = np.minimum(diff.index.dayofyear, 365)
bias_doy = diff.groupby(doy_all).mean()                # (365, 7) — the systematic offset
recomp_c = recomp - bias_doy.loc[doy_all].values
diff_c   = recomp_c - grams


def _summary(d: pd.DataFrame, r: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "MAE":  np.abs(d.values).mean(),
        "RMSE": np.sqrt((d.values ** 2).mean()),
        "max |Δ|": np.abs(d.values).max(),
        "r":    np.corrcoef(r.values.ravel(), grams.loc[r.index].values.ravel())[0, 1],
        "max-IWR match (%)": 100 * (r.idxmax(axis=1) == grams.loc[r.index].idxmax(axis=1)).mean(),
    })


# Out-of-sample check: offset from 1960–1990, applied to 1991–2021
train   = diff.index.year <= 1990
bias_tr = diff[train].groupby(doy_all[train]).mean()
recomp_oos = recomp[~train] - bias_tr.loc[doy_all[~train]].values

bias_tbl = pd.DataFrame({
    "raw":                       _summary(diff, recomp),
    "bias-corrected (in-sample)": _summary(diff_c, recomp_c),
    "raw, 1991–2021":            _summary(diff[~train], recomp[~train]),
    "corrected out-of-sample, 1991–2021": _summary(recomp_oos - grams[~train], recomp_oos),
}).T
print(bias_tbl.round(3).to_string())

per_reg = pd.DataFrame({
    "MAE raw":       diff.abs().mean(),
    "MAE corrected": diff_c.abs().mean(),
    "max |Δ| raw":       diff.abs().max(),
    "max |Δ| corrected": diff_c.abs().max(),
}).T[WR_NAMES]
per_reg["all"] = [np.abs(diff.values).mean(), np.abs(diff_c.values).mean(),
                  np.abs(diff.values).max(), np.abs(diff_c.values).max()]
print()
print(per_reg.round(3).to_string())

share = 1 - (diff_c.values ** 2).mean() / (diff.values ** 2).mean()
print(f"\nShare of the squared error explained by the calendar-day offset: {100 * share:.1f} %")

# %%
mae_doy_c = diff_c.abs().groupby(doy_all).mean()
reg_colors = dict(zip(WR_NAMES, plt.rcParams["axes.prop_cycle"].by_key()["color"]))

fig = plt.figure(figsize=(15, 9))
gs  = fig.add_gridspec(2, 2, width_ratios=[1, 2.2])
ax_bar = fig.add_subplot(gs[:, 0])
ax_b   = fig.add_subplot(gs[0, 1])
ax_r   = fig.add_subplot(gs[1, 1], sharex=ax_b)

x = np.arange(len(WR_NAMES) + 1)
labels = WR_NAMES + ["all"]
ax_bar.barh(x + 0.2, per_reg.loc["MAE raw", labels], 0.4, color="tab:red", label="raw")
ax_bar.barh(x - 0.2, per_reg.loc["MAE corrected", labels], 0.4, color="steelblue",
            label="bias-corrected")
ax_bar.set_yticks(x)
ax_bar.set_yticklabels(labels)
ax_bar.invert_yaxis()
ax_bar.set_xlabel("MAE (recomputed − Grams)")
ax_bar.set_title("MAE per regime, 1960–2021", fontsize=11)
ax_bar.legend(fontsize=9, loc="lower right")
ax_bar.grid(axis="x", color="0.92", lw=0.5)

for reg in WR_NAMES:
    ax_b.plot(bias_doy.index, bias_doy[reg], lw=1.2, color=reg_colors[reg], label=reg)
    ax_r.plot(mae_doy_c.index, mae_doy_c[reg], lw=1.0, color=reg_colors[reg])
ax_b.axhline(0, color="k", lw=0.5)
ax_b.set_ylabel("estimated offset")
ax_b.set_title("Systematic climatology offset per calendar day (subtracted)", fontsize=11)
ax_b.legend(ncol=7, fontsize=8, loc="lower left")
ax_r.set_ylim(0, ax_b.get_ylim()[1] if ax_b.get_ylim()[1] > 0.1 else 0.1)
ax_r.set_ylabel("MAE after correction")
ax_r.set_title("Remaining error per calendar day (same scale as above)", fontsize=11)
ax_r.set_xticks(month_starts.dayofyear)
ax_r.set_xticklabels(month_starts.strftime("%b"))
for ax in (ax_b, ax_r):
    ax.grid(color="0.92", lw=0.5)
fig.tight_layout()
fig.savefig(paths.images_path / "28_iwr_bias_correction.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/28_iwr_bias_correction.png
# :name: fig-28-iwr-bias-correction
# Left: MAE per regime before (red) and after (blue) removing the calendar-day
# offset.  Top right: the offset itself, i.e. the mean recomputed − Grams
# difference per calendar day over 1960–2021.  Bottom right: MAE per calendar
# day after the correction.
# ```

# %% [markdown]
# ### Summer of the representative year after correction
#
# July–September is where the raw error peaks (ScBL, ZO, ScTr, AT, EuBL).

# %%
sl_s = slice(f"{year}-06-01", f"{year}-09-30")
fig, axes = plt.subplots(7, 1, figsize=(14, 16), sharex=True)
for ax, reg in zip(axes, WR_NAMES):
    ax.plot(grams.loc[sl_s].index, grams.loc[sl_s, reg], color="0.3", lw=1.6, label="Grams")
    ax.plot(recomp.loc[sl_s].index, recomp.loc[sl_s, reg], color="tab:red", lw=1.0,
            alpha=0.8, label="recomputed, raw")
    ax.plot(recomp_c.loc[sl_s].index, recomp_c.loc[sl_s, reg], color="steelblue", lw=1.0,
            ls="--", label="recomputed, bias-corrected")
    ax.axhline(0, color="k", lw=0.4)
    ax.axhline(1, color="k", lw=0.4, ls=":")
    ax.set_ylabel(reg)
axes[0].legend(loc="upper right", fontsize=8, ncol=3)
axes[0].set_title(f"IWR June–September {year}: raw vs bias-corrected recomputation", fontsize=11)
fig.tight_layout()
fig.savefig(paths.images_path / "28_iwr_bias_corrected_zoom.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/28_iwr_bias_corrected_zoom.png
# :name: fig-28-iwr-bias-corrected-zoom
# Daily IWR, June–September of the representative year: Grams (grey),
# raw recomputation (red) and after removing the calendar-day offset (blue,
# dashed).
# ```
