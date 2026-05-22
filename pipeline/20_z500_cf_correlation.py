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
# # Z500 Anomaly — Capacity Factor Correlation Maps
#
# For each grid point in the Euro-Atlantic Z500 field, this notebook computes
# the Pearson correlation between the daily Z500 anomaly at that grid point and
# the daily capacity-factor anomaly for:
#
# - **Wind onshore** (Germany)
# - **Solar PV** (Germany)
#
# **Z500 anomaly** = ERA5 daily 12 UTC Z500 minus the WeatherBench2 1979-2019
# day-of-year climatology.
#
# **CF anomaly** = daily mean capacity factor minus the Fourier-smoothed
# day-of-year climatology (4 harmonics, computed from the same 1979–2021
# period used for correlation).
#
# The two datasets overlap over **1979-01-01 → 2021-12-31** (≈ 15 700 days).

# %%
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from wr.paths import ProjPaths

paths = ProjPaths()

PECD_PATH = "/home/chris/research/world-of-energy/data/processed/pecd/pecd_regions.parquet"
IMG_OUT   = paths.images_path / "20_z500_cf_correlation.png"

# %% [markdown]
# ## Load and align Z500

# %%
print("Loading ERA5 Z500 …")
z500_ds = xr.open_zarr(str(paths.era5_z500_daily_zarr))["z500"].load()

print("Loading WB2 Z500 climatology …")
clim = xr.open_zarr(str(paths.wb_z500_climatology))["z500"].load()

# Compute anomaly
doy = z500_ds.time.dt.dayofyear
z500_anom = (z500_ds - clim.sel(dayofyear=doy)).rename("z500_anom")

# Normalise time index to date-only for joining with PECD daily series
z500_dates = pd.to_datetime(z500_anom.time.values).normalize()
z500_anom  = z500_anom.assign_coords(time=z500_dates)

# Filter to PECD overlap period: 1979–2021
z500_anom = z500_anom.sel(time=slice("1979-01-01", "2021-12-31"))
print(f"Z500 anomaly: {str(z500_anom.time.values[0])[:10]} → "
      f"{str(z500_anom.time.values[-1])[:10]}, n={len(z500_anom.time)}")

# %% [markdown]
# ## Load and prepare PECD capacity factors

# %%
print("Loading PECD …")
pecd = pd.read_parquet(PECD_PATH)

de_wind_h  = pecd[("wind_power_generation_onshore",       "capacity_factor_ratio", "DE")]
de_solar_h = pecd[("solar_photovoltaic_power_generation", "capacity_factor_ratio", "DE")]

# Resample to daily means
de_wind_d  = de_wind_h.resample("D").mean()
de_solar_d = de_solar_h.resample("D").mean()

# Filter to same overlap period
de_wind_d  = de_wind_d["1979":"2021"]
de_solar_d = de_solar_d["1979":"2021"]

print(f"Wind  daily: {de_wind_d.index[0].date()} → {de_wind_d.index[-1].date()}, "
      f"n={len(de_wind_d)}, NaN={de_wind_d.isna().sum()}")
print(f"Solar daily: {de_solar_d.index[0].date()} → {de_solar_d.index[-1].date()}, "
      f"n={len(de_solar_d)}, NaN={de_solar_d.isna().sum()}")

# %% [markdown]
# ## Compute CF day-of-year climatology (Fourier, 4 harmonics)

# %%
def fourier_climatology(values: np.ndarray, n_harmonics: int = 4) -> np.ndarray:
    N = len(values)
    t = np.arange(N)
    omega = 2 * np.pi * t / N
    smooth = np.full(N, np.mean(values), dtype=float)
    for h in range(1, n_harmonics + 1):
        cc = np.mean(values * np.cos(h * omega)) * 2
        sc = np.mean(values * np.sin(h * omega)) * 2
        smooth += cc * np.cos(h * omega) + sc * np.sin(h * omega)
    return smooth


def compute_cf_anomaly(series: pd.Series) -> pd.Series:
    """Subtract Fourier-smoothed day-of-year climatology from a daily series."""
    g = series.groupby([series.index.month, series.index.day])
    raw_clim = g.mean()                         # multi-index (month, day)
    smooth   = fourier_climatology(raw_clim.values, n_harmonics=4)

    clim_map = {key: val for key, val in zip(raw_clim.index, smooth)}

    doy_keys = list(zip(series.index.month, series.index.day))
    clim_vals = np.array([clim_map[k] for k in doy_keys])
    return series - clim_vals


wind_anom  = compute_cf_anomaly(de_wind_d)
solar_anom = compute_cf_anomaly(de_solar_d)

print(f"Wind  anom: mean={wind_anom.mean():.4f}, std={wind_anom.std():.4f}")
print(f"Solar anom: mean={solar_anom.mean():.4f}, std={solar_anom.std():.4f}")

# %% [markdown]
# ## Align time indices and compute pointwise correlations

# %%
# Use the intersection of dates present in both datasets
z500_date_index = pd.DatetimeIndex(z500_anom.time.values)
common = z500_date_index.intersection(wind_anom.index)
print(f"Common dates: {common[0].date()} → {common[-1].date()}, n={len(common)}")

z500_sel  = z500_anom.sel(time=common)          # (T, lat, lon)
wind_sel  = wind_anom.reindex(common).values    # (T,)
solar_sel = solar_anom.reindex(common).values   # (T,)

# Vectorised Pearson correlation over the time axis
def pointwise_corr(field: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Pearson r between each (lat, lon) time series and ts.  field: (T, lat, lon)."""
    T = field.shape[0]
    f_mean = field.mean(axis=0)            # (lat, lon)
    t_mean = ts.mean()
    f_c = field - f_mean                   # (T, lat, lon)
    t_c = ts - t_mean                      # (T,)
    cov  = (f_c * t_c[:, None, None]).mean(axis=0)
    f_std = f_c.std(axis=0)
    t_std = t_c.std()
    with np.errstate(invalid="ignore"):
        r = cov / (f_std * t_std)
    return r


print("Computing wind correlation …")
corr_wind  = pointwise_corr(z500_sel.values, wind_sel)

print("Computing solar correlation …")
corr_solar = pointwise_corr(z500_sel.values, solar_sel)

lats = z500_sel.latitude.values
lons = z500_sel.longitude.values

print(f"Wind  corr: min={np.nanmin(corr_wind):.3f}, max={np.nanmax(corr_wind):.3f}")
print(f"Solar corr: min={np.nanmin(corr_solar):.3f}, max={np.nanmax(corr_solar):.3f}")

# %% [markdown]
# ## Correlation maps

# %%
PROJ  = ccrs.PlateCarree()
TRANS = ccrs.PlateCarree()
EXTENT = [-80, 41, 29, 91]

vmax = max(
    np.nanmax(np.abs(corr_wind)),
    np.nanmax(np.abs(corr_solar)),
)
vmax = round(min(vmax, 1.0), 2)
levels = np.linspace(-vmax, vmax, 21)

fig, axes = plt.subplots(
    1, 2, figsize=(16, 6),
    subplot_kw={"projection": PROJ},
    constrained_layout=True,
)

panels = [
    (corr_wind,  "Wind onshore DE",  axes[0]),
    (corr_solar, "Solar PV DE",      axes[1]),
]

for corr, title, ax in panels:
    ax.set_extent(EXTENT, crs=TRANS)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, linestyle=":")
    ax.add_feature(cfeature.LAND,      facecolor="#f5f5f5", zorder=0)
    cf = ax.contourf(
        lons, lats, corr.T,
        levels=levels, cmap="RdBu_r", transform=TRANS, extend="both",
    )
    ax.contour(
        lons, lats, corr.T,
        levels=[0], colors="k", linewidths=0.8, transform=TRANS,
    )
    cb = plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.03, fraction=0.04,
                      label="Pearson r")
    cb.ax.tick_params(labelsize=9)
    ax.set_title(f"Z500 anomaly correlation — {title}", fontsize=12)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.5,
                 x_inline=False, y_inline=False)

fig.suptitle(
    f"Pointwise Z500–CF correlation  ({common[0].year}–{common[-1].year}, daily)",
    fontsize=13, y=1.01,
)

fig.savefig(IMG_OUT, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {IMG_OUT}")

# %% [markdown]
# ```{figure} ../../output/images/20_z500_cf_correlation.png
# :name: fig-20-z500-cf-correlation
# Pearson correlation between the daily Z500 anomaly at each Euro-Atlantic grid
# point and the daily capacity-factor anomaly for Germany wind onshore (left)
# and Germany solar PV (right).  Period: 1979–2021.  Red = positive correlation;
# blue = negative.  The zero-correlation contour is shown in black.
# ```

# %% [markdown]
# ## Lagged and multi-day-average correlation maps
#
# Z500 anomaly at time *t* is correlated with a **w-day forward mean** of the CF
# anomaly starting at *t + lag*:
#
#   CF_avg(t+lag) = mean( CF(t+lag), CF(t+lag+1), …, CF(t+lag+w-1) )
#
# Varying *lag* tests predictive skill; varying *w* tests whether averaging the
# target over multiple days reveals a stronger large-scale signal.
#
# Combinations produced below: windows w ∈ {1, 2, 5} days × lags ∈ {0, 5, 15}
# days (w=1, lag=0 is the simultaneous map already shown above).

# %%
z500_np  = z500_sel.values   # (T, lat, lon)
WINDOWS  = [1, 2, 5]
LAG_DAYS = [0, 5, 15]


def forward_mean(arr: np.ndarray, w: int) -> np.ndarray:
    """Forward w-day mean: result[i] = mean(arr[i:i+w]).  Length = len(arr)-w+1."""
    if w == 1:
        return arr.copy()
    return np.convolve(arr, np.ones(w) / w, mode="valid")


# Pre-compute forward means for all window sizes
wind_fwd  = {w: forward_mean(wind_sel,  w) for w in WINDOWS}
solar_fwd = {w: forward_mean(solar_sel, w) for w in WINDOWS}


def plot_corr_map(corr_w: np.ndarray, corr_s: np.ndarray,
                  w: int, lag: int, n_pairs: int) -> plt.Figure:
    lag_levels = np.linspace(-vmax, vmax, 21)
    if lag == 0:
        lead_str = "simultaneous"
    else:
        lead_str = f"lead {lag} d"
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6),
        subplot_kw={"projection": PROJ},
        constrained_layout=True,
    )
    for corr, var_label, ax in [
        (corr_w, "Wind onshore DE", axes[0]),
        (corr_s, "Solar PV DE",     axes[1]),
    ]:
        ax.set_extent(EXTENT, crs=TRANS)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.4, linestyle=":")
        ax.add_feature(cfeature.LAND,      facecolor="#f5f5f5", zorder=0)
        cf = ax.contourf(
            lons, lats, corr.T,
            levels=lag_levels, cmap="RdBu_r", transform=TRANS, extend="both",
        )
        ax.contour(
            lons, lats, corr.T,
            levels=[0], colors="k", linewidths=0.8, transform=TRANS,
        )
        cb = plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.03,
                          fraction=0.04, label="Pearson r")
        cb.ax.tick_params(labelsize=9)
        ax.set_title(f"Z500(t) → {var_label} {w}d-avg (t+{lag}d)", fontsize=12)
        ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.5,
                     x_inline=False, y_inline=False)
    fig.suptitle(
        f"CF {w}-day average, {lead_str}  "
        f"({common[0].year}–{common[-1].year}, n={n_pairs} pairs)",
        fontsize=13, y=1.01,
    )
    return fig


for w in WINDOWS:
    for lag in LAG_DAYS:
        if w == 1 and lag == 0:
            continue  # already produced as the simultaneous figure above
        cf_wind_w  = wind_fwd[w]    # length T-w+1
        cf_solar_w = solar_fwd[w]   # length T-w+1
        n = len(cf_wind_w) - lag    # valid pairs
        z500_slice  = z500_np[:n]
        wind_slice  = cf_wind_w[lag:lag + n]
        solar_slice = cf_solar_w[lag:lag + n]

        print(f"w={w}d lag={lag:2d}d — computing …", end=" ", flush=True)
        cw = pointwise_corr(z500_slice, wind_slice)
        cs = pointwise_corr(z500_slice, solar_slice)
        print(f"wind max|r|={np.nanmax(np.abs(cw)):.3f}  "
              f"solar max|r|={np.nanmax(np.abs(cs)):.3f}")

        fig  = plot_corr_map(cw, cs, w, lag, n)
        fname = paths.images_path / f"20_z500_cf_corr_w{w}d_lag{lag:02d}d.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {fname.name}")

# %% [markdown]
# ## 1-day CF average (instantaneous)
#
# ### Lead 5 days — 1-day avg
#
# ```{figure} ../../output/images/20_z500_cf_corr_w1d_lag05d.png
# :name: fig-20-z500-cf-corr-w1d-lag05d
# Z500(t) correlated with 1-day CF anomaly at t+5 d.
# ```
#
# ### Lead 15 days — 1-day avg
#
# ```{figure} ../../output/images/20_z500_cf_corr_w1d_lag15d.png
# :name: fig-20-z500-cf-corr-w1d-lag15d
# Z500(t) correlated with 1-day CF anomaly at t+15 d.
# ```

# %% [markdown]
# ## 2-day CF average
#
# ### Simultaneous — 2-day avg
#
# ```{figure} ../../output/images/20_z500_cf_corr_w2d_lag00d.png
# :name: fig-20-z500-cf-corr-w2d-lag00d
# Z500(t) correlated with 2-day CF average starting at t.
# ```
#
# ### Lead 5 days — 2-day avg
#
# ```{figure} ../../output/images/20_z500_cf_corr_w2d_lag05d.png
# :name: fig-20-z500-cf-corr-w2d-lag05d
# Z500(t) correlated with 2-day CF average starting at t+5 d.
# ```
#
# ### Lead 15 days — 2-day avg
#
# ```{figure} ../../output/images/20_z500_cf_corr_w2d_lag15d.png
# :name: fig-20-z500-cf-corr-w2d-lag15d
# Z500(t) correlated with 2-day CF average starting at t+15 d.
# ```

# %% [markdown]
# ## 5-day CF average
#
# ### Simultaneous — 5-day avg
#
# ```{figure} ../../output/images/20_z500_cf_corr_w5d_lag00d.png
# :name: fig-20-z500-cf-corr-w5d-lag00d
# Z500(t) correlated with 5-day CF average starting at t.
# ```
#
# ### Lead 5 days — 5-day avg
#
# ```{figure} ../../output/images/20_z500_cf_corr_w5d_lag05d.png
# :name: fig-20-z500-cf-corr-w5d-lag05d
# Z500(t) correlated with 5-day CF average starting at t+5 d.
# ```
#
# ### Lead 15 days — 5-day avg
#
# ```{figure} ../../output/images/20_z500_cf_corr_w5d_lag15d.png
# :name: fig-20-z500-cf-corr-w5d-lag15d
# Z500(t) correlated with 5-day CF average starting at t+15 d.
# ```
