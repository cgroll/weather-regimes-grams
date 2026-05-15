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
# # Weather Regime Index Time Series
#
# Recreates the IWR time series plot from Grams et al. (2025) for a
# selectable period. Thick lines show the WR index only during active life
# cycles; thin lines show the full (always-computed) index. A colour strip at
# the bottom encodes the life-cycle attribution regime at each time step.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wr.paths import ProjPaths
from wr.regimes import REGIMES, BY_INDEX, WR_NAMES

paths = ProjPaths()

# %%
wri      = pd.read_csv(paths.wri_csv,          parse_dates=["datetime"], index_col="datetime")
lc_info  = pd.read_csv(paths.lc_info_csv,      parse_dates=["onset", "decay"])
lc_attr  = pd.read_csv(paths.lc_attribution_csv, parse_dates=["datetime"], index_col="datetime")

# %% [markdown]
# ## Period selection

# %%
START = "2024-11-01"
END   = "2025-03-31"

wri_p    = wri.loc[START:END]
lc_attr_p = lc_attr.loc[START:END]

# %% [markdown]
# ## Build active-lifecycle IWR mask
#
# For each regime, keep IWR values only during active life cycles
# (onset ≤ t < decay); everything else becomes NaN (shown as thin line only).

# %%
t_start = pd.Timestamp(START)
t_end   = pd.Timestamp(END)

wri_active = pd.DataFrame(np.nan, index=wri_p.index, columns=WR_NAMES)

for regime in WR_NAMES:
    active_lcs = lc_info[
        (lc_info["regime"] == regime) &
        (lc_info["decay"]  > t_start) &
        (lc_info["onset"]  < t_end)
    ]
    for _, lc in active_lcs.iterrows():
        mask = (wri_p.index >= lc["onset"]) & (wri_p.index < lc["decay"])
        wri_active.loc[mask, regime] = wri_p.loc[mask, regime]

# %% [markdown]
# ## Plot

# %%
COLORS = {r["name"]: r["color"] for r in REGIMES}
ymin, ymax = -4, 4

fig, ax = plt.subplots(figsize=(14, 5))

# Thin lines — full IWR (no label to avoid duplicate legend entries)
for regime in WR_NAMES:
    ax.plot(wri_p.index, wri_p[regime], color=COLORS[regime], linewidth=1.0)

# Thick lines — active life-cycle IWR (carry the legend labels)
for regime in WR_NAMES:
    ax.plot(wri_p.index, wri_active[regime], color=COLORS[regime], linewidth=3.0, label=regime)

ax.legend(loc="upper right", ncol=3, fontsize=12)

# Reference lines
ax.axhline(0, color="black", linewidth=1.0)
ax.axhline(1, color="black", linewidth=0.5)

ax.set_ylim(ymin, ymax)
ax.set_xlim(wri_p.index[0], wri_p.index[-1])
ax.set_ylabel("Weather regime index (IWR)", fontsize=15)
ax.set_xlabel("time", fontsize=15)
ax.tick_params(axis="both", which="major", labelsize=13)

# LC attribution colour strip at the bottom
lc_idx = lc_attr_p["lifecycle_wr_index"]
blocks = (lc_idx != lc_idx.shift()).cumsum()
for _, group in lc_idx.groupby(blocks):
    ax.fill_between(
        [group.index[0], group.index[-1]], ymin, ymin + 0.2,
        color=BY_INDEX[group.iloc[0]]["color"],
    )

# Month ticks and vertical dashed lines
month_ticks, month_labels = [], []
for ts in wri_p.index:
    if ts.day == 1 and ts.hour == 0:
        month_ticks.append(ts)
        month_labels.append(ts.strftime("1 %b\n%Y"))
        ax.axvline(ts, color="black", linestyle="dashed", linewidth=0.5)

ax.set_xticks(month_ticks)
ax.set_xticklabels(month_labels, fontsize=13)

fig.savefig(paths.images_path / "04_wr_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/04_wr_timeseries.png
# :name: fig-04-wr-timeseries
# Weather regime index (IWR) for all 7 regimes, Nov 2024 – Mar 2025. Thick
# lines mark active life cycles; thin lines show the full (always-computed)
# index. The colour strip at the bottom encodes the life-cycle attribution
# regime at each 3-hourly time step.
# ```

# %% [markdown]
# ## Regime frequency over the full period

# %%
freq = lc_attr["lifecycle_wr_index"].value_counts().sort_index()
freq_pct = freq / freq.sum() * 100

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(
    [BY_INDEX[i]["name"]  for i in freq_pct.index],
    freq_pct.values,
    color=[BY_INDEX[i]["color"] for i in freq_pct.index],
)
ax.set_ylabel("Frequency (%)", fontsize=13)
ax.set_xlabel("Regime", fontsize=13)
ax.tick_params(labelsize=12)
fig.tight_layout()
fig.savefig(paths.images_path / "04_wr_freq_overall.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/04_wr_freq_overall.png
# :name: fig-04-wr-freq-overall
# Regime frequency over the full 1950–2025 period based on 3-hourly
# lifecycle attribution.
# ```

# %% [markdown]
# ## Annual regime frequencies (stacked bar)

# %%
yearly_counts = (
    lc_attr.assign(year=lc_attr.index.year)
    .groupby(["year", "lifecycle_wr_index"]).size()
    .unstack(fill_value=0)
)
yearly_pct = yearly_counts.div(yearly_counts.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(16, 5))
years  = yearly_pct.index.values
bottom = np.zeros(len(years))
for idx in range(8):
    if idx not in yearly_pct.columns:
        continue
    vals = yearly_pct[idx].values
    ax.bar(years, vals, bottom=bottom,
           color=BY_INDEX[idx]["color"], label=BY_INDEX[idx]["name"], width=0.8)
    bottom += vals

ax.set_ylim(0, 100)
ax.set_ylabel("Frequency (%)", fontsize=13)
ax.set_xlabel("Year", fontsize=13)
ax.set_xticks(range(1950, 2026, 5))
ax.tick_params(labelsize=11)
ax.legend(loc="lower left", ncol=4, fontsize=11)
fig.tight_layout()
fig.savefig(paths.images_path / "04_wr_freq_annual.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/04_wr_freq_annual.png
# :name: fig-04-wr-freq-annual
# Annual regime frequency 1950–2025. Each bar sums to 100 %; partial years
# (1950 starts 11 Jan, 2025 ends 26 Jul) still sum to 100 % within the year.
# ```

# %% [markdown]
# ## Calendar view: regime by year and day of year

# %%
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# Daily representative: 12 UTC value
daily = lc_attr.loc[lc_attr.index.hour == 12, "lifecycle_wr_index"]
daily_df = pd.DataFrame({
    "year":   daily.index.year,
    "doy":    daily.index.dayofyear,
    "regime": daily.values,
})
pivot = daily_df.pivot_table(index="year", columns="doy", values="regime", aggfunc="first")

cal_cmap = ListedColormap([BY_INDEX[i]["color"] for i in range(8)])
bounds   = np.arange(-0.5, 8.5)
norm     = BoundaryNorm(bounds, cal_cmap.N)
cal_cmap.set_bad("white")  # leap-day gaps and incomplete years

data = np.ma.masked_invalid(pivot.values.astype(float))

fig, ax = plt.subplots(figsize=(18, 10))
ax.pcolormesh(pivot.columns.values, pivot.index.values, data,
              cmap=cal_cmap, norm=norm, shading="nearest")

# Month labels at the 1st of each month (non-leap DOY)
MONTH_DOY   = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ax.set_xticks(MONTH_DOY)
ax.set_xticklabels(MONTH_NAMES, fontsize=12)
ax.set_xlim(1, 366)

ax.set_ylabel("Year", fontsize=13)
ax.set_yticks(range(1950, 2026, 5))
ax.tick_params(axis="y", labelsize=11)
ax.invert_yaxis()  # 1950 at top

handles = [mpatches.Patch(color=BY_INDEX[i]["color"], label=BY_INDEX[i]["name"])
           for i in range(8)]
ax.legend(handles=handles, loc="lower right", ncol=4, fontsize=11, framealpha=0.9)

fig.tight_layout()
fig.savefig(paths.images_path / "04_wr_calendar.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/04_wr_calendar.png
# :name: fig-04-wr-calendar
# Calendar view of the lifecycle-attributed regime at 12 UTC for each day,
# 1950–2025. Each row is one year; columns are day of year. White cells are
# missing (leap day in non-leap years, or outside the dataset period).
# ```
