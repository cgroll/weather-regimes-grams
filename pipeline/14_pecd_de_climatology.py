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
# # PECD Germany: Daily Capacity Factor Climatology
#
# Computes the day-of-year climatology of German wind onshore and solar PV
# capacity factors and compares three representations:
#
# - **Raw**: simple mean across all years for each calendar day
# - **Rolling window**: 31-day circular rolling mean of the raw daily means
# - **Fourier**: reconstruction using the first N annual harmonics
#
# ## Methodological notes
#
# ### Why group by (month, day) rather than dayofyear?
#
# `pandas.DatetimeIndex.dayofyear` assigns doy=60 to *Feb 29* in leap years
# and to *Mar 1* in non-leap years. Grouping by doy therefore mixes two
# different calendar dates for all days from Mar 1 onward.  Grouping by
# `(month, day)` always aligns the same date regardless of leap-year status.
#
# ### Feb 29
#
# Feb 29 has roughly one-quarter of the sample size of other days (~12 years
# vs ~46 years). It is included as a real observation; its wider uncertainty
# is visible in the raw scatter.
#
# ### Fourier harmonics
#
# The Fourier fit reconstructs the seasonal cycle from its first N harmonics
# (annual, semi-annual, …).  The 366-day series (including Feb 29) is used
# directly; the period is set to N=366, which is consistent with treating
# the climatological year as 366 equally-spaced days.  For wind in Germany,
# harmonics 1–2 capture most of the variance; additional harmonics refine the
# shoulder-season shape.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from wr.paths import ProjPaths

paths = ProjPaths()

PECD_PATH = "/home/chris/research/world-of-energy/data/processed/pecd/pecd_regions.parquet"
IMG_OUT   = paths.images_path / "14_pecd_de_climatology.png"

# %%
pecd = pd.read_parquet(PECD_PATH)

de_wind  = pecd[("wind_power_generation_onshore",       "capacity_factor_ratio", "DE")]
de_solar = pecd[("solar_photovoltaic_power_generation", "capacity_factor_ratio", "DE")]

# Resample to daily means so each year-day counts as one observation.
# On hourly data, groupby(...).mean() gives the same point estimate, but
# std and count become meaningful only after daily resampling.
de_wind_d  = de_wind.resample("D").mean()
de_solar_d = de_solar.resample("D").mean()

print(f"Wind  daily: {de_wind_d.index.min().date()} – {de_wind_d.index.max().date()}, "
      f"N={len(de_wind_d)}, NaN={de_wind_d.isna().sum()}")
print(f"Solar daily: {de_solar_d.index.min().date()} – {de_solar_d.index.max().date()}, "
      f"N={len(de_solar_d)}, NaN={de_solar_d.isna().sum()}")

# %% [markdown]
# ## Climatology functions

# %%
def daily_climatology(series: pd.Series) -> pd.DataFrame:
    """Group by (month, day), return mean / std / count per calendar day."""
    g = series.groupby([series.index.month, series.index.day])
    clim = g.agg(["mean", "std", "count"])
    clim.index.names = ["month", "day"]
    return clim


def rolling_climatology(values: np.ndarray, window: int = 31) -> np.ndarray:
    """
    Circular rolling window mean over a daily-climatology array.

    Wraps the array at both ends before rolling so Jan and Dec are smoothed
    consistently with the rest of the year — without padding, the rolling
    window would see zeros (or NaNs) at the boundaries instead of the
    actual Dec / Jan values.
    """
    pad = window // 2
    s = pd.Series(values)
    padded = pd.concat([s.tail(pad), s, s.head(pad)], ignore_index=True)
    smoothed = padded.rolling(window=window, center=True, min_periods=1).mean()
    return smoothed.iloc[pad : pad + len(values)].values


def fourier_climatology(values: np.ndarray, n_harmonics: int = 4) -> np.ndarray:
    """
    Reconstruct daily climatology retaining only the first n_harmonics
    annual harmonics plus the annual mean.

    N is derived from the length of the input (365 or 366), so the period
    naturally adapts to whether Feb 29 is included.
    """
    N = len(values)
    t = np.arange(N)
    omega = 2 * np.pi * t / N
    smooth = np.full(N, np.mean(values), dtype=float)
    for h in range(1, n_harmonics + 1):
        cc = np.mean(values * np.cos(h * omega)) * 2
        sc = np.mean(values * np.sin(h * omega)) * 2
        smooth += cc * np.cos(h * omega) + sc * np.sin(h * omega)
    return smooth


# Use a leap year as the reference calendar so Feb 29 has a real date.
REF_YEAR = 2000


def clim_dates(clim_df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.to_datetime(
        [f"{REF_YEAR}-{m:02d}-{d:02d}" for m, d in clim_df.index]
    )


# %% [markdown]
# ## Compute climatologies

# %%
wind_clim  = daily_climatology(de_wind_d)
solar_clim = daily_climatology(de_solar_d)

for clim, label in [(wind_clim, "wind "), (solar_clim, "solar")]:
    row = clim.loc[(2, 29)]
    print(f"{label}  Feb 29: n={row['count']:.0f}, mean={row['mean']:.3f}")

# %% [markdown]
# ## Comparison chart

# %%
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

specs = [
    (wind_clim,  "Wind onshore",  "steelblue"),
    (solar_clim, "Solar PV",      "darkorange"),
]

for ax, (clim, title, color) in zip(axes, specs):
    dates = clim_dates(clim)
    raw   = clim["mean"].values
    roll  = rolling_climatology(raw, window=31)
    four  = fourier_climatology(raw, n_harmonics=4)

    ax.plot(dates, raw,  color="grey",  lw=0.8, alpha=0.55, label="Raw daily mean")
    ax.plot(dates, roll, color=color,   lw=2.0,              label="Rolling window (31 d)")
    ax.plot(dates, four, color="black", lw=1.5, ls="--",    label="Fourier (4 harmonics)")

    feb29_date = pd.Timestamp(f"{REF_YEAR}-02-29")
    feb29_val  = clim.loc[(2, 29), "mean"]
    ax.axvline(feb29_date, color="red", lw=0.8, alpha=0.35, ls=":")
    ax.scatter([feb29_date], [feb29_val], color="red", s=35, zorder=5,
               label=f"Feb 29  (n={clim.loc[(2,29),'count']:.0f} yr)")

    ax.set_ylabel("Capacity factor", fontsize=11)
    ax.set_title(f"Germany {title} — Daily climatology (1979–2025)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right" if title == "Wind onshore" else "upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axes[-1].tick_params(axis="x", labelsize=10)

fig.tight_layout()
fig.savefig(IMG_OUT, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {IMG_OUT}")

# %% [markdown]
# ```{figure} ../../output/images/14_pecd_de_climatology.png
# :name: fig-14-pecd-de-climatology
# Germany wind onshore (top) and solar PV (bottom) daily capacity factor
# climatology (1979–2025).  Grey: raw day-of-year means; coloured line:
# 31-day circular rolling window; dashed black: 4-harmonic Fourier
# reconstruction.  Feb 29 is marked in red (12 sample years only).
# ```
