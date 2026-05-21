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
# # PECD Energy Data Overview
#
# Exploratory overview of the Pan-European Climate Database (PECD) energy
# dataset from the `world-of-energy` project. This is the first step toward
# linking European weather regimes to renewable power generation.
#
# The dataset provides hourly, country-level time series of:
# - Solar PV capacity factor ratio
# - Wind onshore / offshore capacity factor ratio
# - 2 m air temperature
# - Surface downwelling shortwave radiation
# - 10 m wind speed
#
# covering **38–39 European countries** from **1979-01-01 to 2026-01-31**.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from wr.paths import ProjPaths

paths = ProjPaths()

PECD_PATH = "/home/chris/research/world-of-energy/data/processed/pecd/pecd_regions.parquet"

# %% [markdown]
# ## Load data

# %%
df = pd.read_parquet(PECD_PATH)

# Flatten multi-level column names for easier indexing
df.columns.names = ["variable", "product_type", "country"]

print("Shape:", df.shape)
print("Time range:", df.index.min(), "→", df.index.max())
print("Temporal resolution:", pd.infer_freq(df.index[:1000]))

# %% [markdown]
# ## Dataset structure

# %%
col_frame = df.columns.to_frame(index=False)
summary = (
    col_frame
    .groupby(["variable", "product_type"])["country"]
    .apply(lambda x: sorted(x.tolist()))
    .reset_index()
)
summary["n_countries"] = summary["country"].apply(len)
print(summary[["variable", "product_type", "n_countries"]].to_string(index=False))

# %% [markdown]
# ### Countries covered

# %%
all_countries = sorted(set(col_frame["country"]))
print(f"{len(all_countries)} countries:", all_countries)

# %% [markdown]
# ## Capacity factor time series — selected countries
#
# Wind onshore and solar PV for DE, FR, UK, ES over a recent full year (2023).

# %%
YEAR = 2023
FOCUS = ["DE", "FR", "UK", "ES"]
COLORS = {"DE": "#2196F3", "FR": "#F44336", "UK": "#4CAF50", "ES": "#FF9800"}

solar = df["solar_photovoltaic_power_generation"]["capacity_factor_ratio"][FOCUS]
wind  = df["wind_power_generation_onshore"]["capacity_factor_ratio"][FOCUS]

# Resample to daily means for readability
solar_d = solar.resample("D").mean().loc[str(YEAR)]
wind_d  = wind.resample("D").mean().loc[str(YEAR)]

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

for country in FOCUS:
    axes[0].plot(solar_d.index, solar_d[country], label=country,
                 color=COLORS[country], linewidth=1.0, alpha=0.85)
    axes[1].plot(wind_d.index,  wind_d[country],  label=country,
                 color=COLORS[country], linewidth=1.0, alpha=0.85)

axes[0].set_ylabel("Solar PV CF (daily mean)", fontsize=12)
axes[1].set_ylabel("Wind onshore CF (daily mean)", fontsize=12)
axes[0].legend(ncol=4, fontsize=11, loc="upper right")
axes[1].legend(ncol=4, fontsize=11, loc="upper right")
for ax in axes:
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=11)

axes[1].set_xlabel(str(YEAR), fontsize=12)
fig.suptitle(f"Daily-mean capacity factors — {YEAR}", fontsize=14)
fig.tight_layout()
fig.savefig(paths.images_path / "09_pecd_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/09_pecd_timeseries.png
# :name: fig-09-pecd-timeseries
# Daily-mean solar PV (top) and wind onshore (bottom) capacity factor ratios for
# DE, FR, UK, and ES in 2023. Solar shows a clear seasonal envelope; wind is more
# volatile but stronger in winter.
# ```

# %% [markdown]
# ## Seasonal profiles (monthly means, 1979–2025)

# %%
# Use full overlap period with Grams WR data
solar_full = df["solar_photovoltaic_power_generation"]["capacity_factor_ratio"][FOCUS]
wind_full  = df["wind_power_generation_onshore"]["capacity_factor_ratio"][FOCUS]

solar_monthly = solar_full.groupby(solar_full.index.month).mean()
wind_monthly  = wind_full.groupby(wind_full.index.month).mean()

month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

for country in FOCUS:
    axes[0].plot(range(1, 13), solar_monthly[country], marker="o", markersize=4,
                 label=country, color=COLORS[country])
    axes[1].plot(range(1, 13), wind_monthly[country],  marker="o", markersize=4,
                 label=country, color=COLORS[country])

for ax, title in zip(axes, ["Solar PV capacity factor", "Wind onshore capacity factor"]):
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, fontsize=10)
    ax.set_ylabel("Mean CF", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=10)

fig.suptitle("Monthly-mean capacity factors 1979–2025", fontsize=13)
fig.tight_layout()
fig.savefig(paths.images_path / "09_pecd_seasonal.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/09_pecd_seasonal.png
# :name: fig-09-pecd-seasonal
# Climatological monthly-mean capacity factors (1979–2025) for solar PV (left)
# and wind onshore (right). Solar peaks in summer; wind in winter — the two
# sources are anti-correlated seasonally, which motivates linking them to
# weather regimes.
# ```

# %% [markdown]
# ## Country comparison — mean capacity factors

# %%
# Compute long-run mean CF across all available countries
solar_all   = df["solar_photovoltaic_power_generation"]["capacity_factor_ratio"]
wind_on_all = df["wind_power_generation_onshore"]["capacity_factor_ratio"]
wind_off_all = df["wind_power_generation_offshore"]["capacity_factor_ratio"]

solar_mean   = solar_all.mean().sort_values(ascending=False)
wind_on_mean = wind_on_all.mean().sort_values(ascending=False)
wind_off_mean = wind_off_all.mean().sort_values(ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, mean_series, title in zip(
    axes,
    [solar_mean, wind_on_mean, wind_off_mean],
    ["Solar PV", "Wind onshore", "Wind offshore"],
):
    ax.barh(mean_series.index, mean_series.values, color="#4C72B0")
    ax.set_xlabel("Mean CF", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=9)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

fig.suptitle("Long-run mean capacity factor by country (1979–2025)", fontsize=13)
fig.tight_layout()
fig.savefig(paths.images_path / "09_pecd_country_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/09_pecd_country_comparison.png
# :name: fig-09-pecd-country-comparison
# Long-run (1979–2025) mean capacity factors for solar PV (left), wind onshore
# (centre), and wind offshore (right) across all available European countries.
# Southern countries (ES, PT, IT) lead on solar; northern and island countries
# (UK, IE, DK, NO) lead on wind.
# ```

# %% [markdown]
# ## Data coverage summary

# %%
# Fraction of non-NaN values per variable / country
coverage = pd.DataFrame({
    "solar":     solar_all.notna().mean(),
    "wind_on":   wind_on_all.notna().mean(),
    "wind_off":  wind_off_all.reindex(columns=wind_on_all.columns).notna().mean()
    if hasattr(wind_off_all, "columns") else np.nan,
})

print("Data coverage (fraction non-NaN):")
print(coverage.to_string())
