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
# # Low CF Events — Germany
#
# Identifies the worst 2-day aggregate capacity factor event in Germany for
# each generation type (solar PV, wind onshore, wind offshore) and inspects
# the surrounding ±10 day window.
#
# For each event:
# 1. **CF time series** — hourly actual CF vs hourly day-of-year climatology
# 2. **WR indices** — 3-hourly IWR for all 7 regimes
# 3. **Regime attribution** — dominant lifecycle regime per day

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

from wr.paths import ProjPaths
from wr.regimes import BY_INDEX, WR_NAMES

paths = ProjPaths()

PECD_PATH = "/home/chris/research/world-of-energy/data/processed/pecd/pecd_regions.parquet"

# %% [markdown]
# ## Load data

# %%
pecd = pd.read_parquet(PECD_PATH)

# Raw hourly CF for Germany — used for panel 1 visualisation
de_solar_h    = pecd[("solar_photovoltaic_power_generation",  "capacity_factor_ratio", "DE")]
de_wind_on_h  = pecd[("wind_power_generation_onshore",        "capacity_factor_ratio", "DE")]
de_wind_off_h = pecd[("wind_power_generation_offshore",       "capacity_factor_ratio", "DE")]

# Daily means — used only for worst-event detection
de_solar_d    = de_solar_h.resample("D").mean()
de_wind_on_d  = de_wind_on_h.resample("D").mean()
de_wind_off_d = de_wind_off_h.resample("D").mean()

# WRI: 3-hourly (native resolution)
wri_3h = pd.read_csv(paths.wri_csv, parse_dates=["datetime"], index_col="datetime")

# LC attribution: 3-hourly → daily dominant regime
lc = pd.read_csv(paths.lc_attribution_csv, parse_dates=["datetime"], index_col="datetime")
lc_daily = (
    lc["lifecycle_wr_index"]
    .resample("D")
    .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
    .astype(float)
)

# %% [markdown]
# ## Hourly day-of-year climatology (1979–2025)
#
# Mean CF for every (day-of-year, hour) combination — preserves the diurnal cycle.

# %%
def hourly_doy_clim(series: pd.Series) -> pd.Series:
    """Return Series indexed by (dayofyear, hour) → mean CF."""
    return series.groupby([series.index.dayofyear, series.index.hour]).mean()

solar_clim    = hourly_doy_clim(de_solar_h)
wind_on_clim  = hourly_doy_clim(de_wind_on_h)
wind_off_clim = hourly_doy_clim(de_wind_off_h)

# %% [markdown]
# ## Worst 2-day events per variable

# %%
def find_worst_2day(daily: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) dates of the consecutive 2-day window with lowest mean CF."""
    end   = daily.rolling(2, min_periods=2).mean().idxmin()
    start = end - pd.Timedelta(days=1)
    return start, end

events = {
    "Solar PV CF":      (de_solar_h,    solar_clim,    *find_worst_2day(de_solar_d)),
    "Wind onshore CF":  (de_wind_on_h,  wind_on_clim,  *find_worst_2day(de_wind_on_d)),
    "Wind offshore CF": (de_wind_off_h, wind_off_clim, *find_worst_2day(de_wind_off_d)),
}

for label, (_, _, ev_s, ev_e) in events.items():
    print(f"{label:<22}  worst window: {ev_s.date()} – {ev_e.date()}")

# %% [markdown]
# ## Event window plots

# %%
WR_COLORS = {BY_INDEX[i]["name"]: BY_INDEX[i]["color"] for i in range(8)}
WINDOW = 10   # days before/after the 2-day event


def plot_event_window(
    cf_h: pd.Series,
    cf_clim: pd.Series,
    var_label: str,
    ev_start: pd.Timestamp,
    ev_end: pd.Timestamp,
) -> plt.Figure:
    win_start = ev_start - pd.Timedelta(days=WINDOW)
    win_end   = ev_end   + pd.Timedelta(days=WINDOW)

    # Hourly index for CF panel
    hours = pd.date_range(win_start, win_end + pd.Timedelta(hours=23), freq="h")
    cf_win = cf_h.reindex(hours)

    # Hourly DOY climatology lookup
    clim_win = pd.Series(
        [cf_clim.get((d.dayofyear if d.dayofyear <= 365 else 365, d.hour), np.nan)
         for d in hours],
        index=hours,
    )

    # 3-hourly WRI for the window
    wri_win = wri_3h.loc[win_start : win_end + pd.Timedelta(hours=23)]

    # Daily LC attribution
    lc_win = lc_daily.loc[win_start : win_end]

    # Vertical lines marking the 2-day event
    line_left  = ev_start
    line_right = ev_end + pd.Timedelta(days=1)   # midnight after end of second day

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 9),
        gridspec_kw={"height_ratios": [3, 3, 1]},
        sharex=True,
    )

    # ── Panel 1: CF vs climatology (hourly) ──────────────────────────────
    ax = axes[0]
    ax.plot(hours, clim_win, color="black",     linewidth=1.0, linestyle="--",
            label="DOY mean (climatology)", zorder=3)
    ax.plot(hours, cf_win,   color="steelblue", linewidth=1.2,
            label=var_label, zorder=4)
    ax.axvline(line_left,  color="red", linewidth=1.5, linestyle="--", zorder=5)
    ax.axvline(line_right, color="red", linewidth=1.5, linestyle="--", zorder=5)
    ax.set_ylabel(var_label, fontsize=12)
    ax.set_ylim(0, None)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.tick_params(labelsize=10)

    # ── Panel 2: WR indices (3-hourly) ───────────────────────────────────
    ax = axes[1]
    for regime in WR_NAMES:
        ax.plot(wri_win.index, wri_win[regime], color=WR_COLORS[regime],
                linewidth=1.4, label=regime)
    ax.axhline(0,  color="black", linewidth=0.8)
    ax.axhline(1,  color="black", linewidth=0.5, linestyle=":")
    ax.axhline(-1, color="black", linewidth=0.5, linestyle=":")
    ax.axvline(line_left,  color="red", linewidth=1.5, linestyle="--", zorder=5)
    ax.axvline(line_right, color="red", linewidth=1.5, linestyle="--", zorder=5)
    ax.set_ylabel("WR index (IWR)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left", ncol=7)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.tick_params(labelsize=10)

    # ── Panel 3: Regime attribution (daily) ──────────────────────────────
    ax = axes[2]
    for d, regime_idx in lc_win.items():
        if pd.notna(regime_idx):
            color = BY_INDEX[int(regime_idx)]["color"]
            ax.axvspan(d, d + pd.Timedelta(days=1), ymin=0, ymax=1,
                       color=color, alpha=0.85)
    ax.axvline(line_left,  color="red", linewidth=1.5, linestyle="--", zorder=5)
    ax.axvline(line_right, color="red", linewidth=1.5, linestyle="--", zorder=5)
    ax.set_yticks([])
    ax.set_ylabel("Attribution", fontsize=10, labelpad=14)
    handles = [mpatches.Patch(color=BY_INDEX[i]["color"], label=BY_INDEX[i]["name"])
               for i in range(8)]
    ax.legend(handles=handles, loc="upper left", ncol=8, fontsize=8, framealpha=0.9)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
    ax.set_xlim(hours[0], hours[-1])
    ax.tick_params(axis="x", labelsize=9)

    fig.suptitle(
        f"Worst 2-day {var_label} event — Germany\n"
        f"{ev_start.date()} – {ev_end.date()}  (±{WINDOW} day window)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig


# %% [markdown]
# ## Solar PV — worst 2-day event

# %%
label, (cf_h, clim, ev_s, ev_e) = list(events.items())[0]
fig = plot_event_window(cf_h, clim, label, ev_s, ev_e)
fig.savefig(paths.images_path / "12_low_cf_worst_solar.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/12_low_cf_worst_solar.png
# :name: fig-12-low-cf-worst-solar
# Worst 2-day solar PV CF event in Germany. Top: hourly CF (blue) vs hourly
# DOY climatology (dashed). Middle: 3-hourly WR indices. Bottom: daily lifecycle
# regime attribution. Red dashed lines bracket the 2-day event.
# ```

# %% [markdown]
# ## Wind onshore — worst 2-day event

# %%
label, (cf_h, clim, ev_s, ev_e) = list(events.items())[1]
fig = plot_event_window(cf_h, clim, label, ev_s, ev_e)
fig.savefig(paths.images_path / "12_low_cf_worst_wind_on.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/12_low_cf_worst_wind_on.png
# :name: fig-12-low-cf-worst-wind-on
# Worst 2-day wind onshore CF event in Germany. Layout as above.
# ```

# %% [markdown]
# ## Wind offshore — worst 2-day event

# %%
label, (cf_h, clim, ev_s, ev_e) = list(events.items())[2]
fig = plot_event_window(cf_h, clim, label, ev_s, ev_e)
fig.savefig(paths.images_path / "12_low_cf_worst_wind_off.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/12_low_cf_worst_wind_off.png
# :name: fig-12-low-cf-worst-wind-off
# Worst 2-day wind offshore CF event in Germany. Layout as above.
# ```
