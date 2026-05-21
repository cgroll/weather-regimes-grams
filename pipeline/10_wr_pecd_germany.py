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
# # Weather Regimes and Power Generation — Germany
#
# Links the Grams weather regime indices and lifecycle attribution to PECD
# capacity factors for Germany (DE).
#
# **Variables for Germany:**
# - Solar PV capacity factor ratio
# - Wind onshore capacity factor ratio
# - Wind offshore capacity factor ratio
#
# **Weather regime data (Grams):**
# - `WRI_projections.csv` — 3-hourly WR index for 7 regimes, 1950–2025
# - `lc_attribution.csv` — 3-hourly lifecycle attribution (regime 0–7 or
#   "no regime" = 0)
#
# All series are resampled to **daily** resolution before analysis to suppress
# the diurnal cycle and align the two datasets.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from wr.paths import ProjPaths
from wr.regimes import BY_INDEX, WR_NAMES

paths = ProjPaths()

PECD_PATH = "/home/chris/research/world-of-energy/data/processed/pecd/pecd_regions.parquet"

# %% [markdown]
# ## Load and align data

# %%
# --- PECD: Germany CFs (hourly) ---
pecd = pd.read_parquet(PECD_PATH)
de_solar    = pecd[("solar_photovoltaic_power_generation",           "capacity_factor_ratio", "DE")]
de_wind_on  = pecd[("wind_power_generation_onshore",                 "capacity_factor_ratio", "DE")]
de_wind_off = pecd[("wind_power_generation_offshore",                "capacity_factor_ratio", "DE")]
de_temp     = pecd[("2m_air_temperature",                            "value",                 "DE")]
de_rad      = pecd[("surface_downwelling_shortwave_radiation",        "value",                 "DE")]
de_wspeed   = pecd[("wind_speed_at_10m",                             "value",                 "DE")]

# Convert temperature from Kelvin to Celsius
de_temp = de_temp - 273.15

# Resample to daily means (solar: NaN at night are skipped, giving daytime mean)
de_daily = pd.DataFrame({
    "solar":    de_solar.resample("D").mean(),
    "wind_on":  de_wind_on.resample("D").mean(),
    "wind_off": de_wind_off.resample("D").mean(),
    "temp":     de_temp.resample("D").mean(),
    "rad":      de_rad.resample("D").mean(),
    "wspeed":   de_wspeed.resample("D").mean(),
})

# --- WRI: 3-hourly → daily mean ---
wri = pd.read_csv(paths.wri_csv, parse_dates=["datetime"], index_col="datetime")
wri_daily = wri.resample("D").mean()

# --- LC attribution: 3-hourly → daily mode (dominant regime) ---
lc = pd.read_csv(paths.lc_attribution_csv, parse_dates=["datetime"], index_col="datetime")
lc_daily = (
    lc["lifecycle_wr_index"]
    .resample("D")
    .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
    .astype(float)
)

# --- Align on inner join: overlap is 1979–2025 ---
data = (
    wri_daily
    .join(de_daily, how="inner")
    .join(lc_daily.rename("regime"), how="inner")
    .dropna(subset=["wind_on", "wind_off"])   # solar may still have NaN; kept as-is
)

data["regime"] = data["regime"].astype(int)

print("Aligned dataset shape:", data.shape)
print("Date range:", data.index.min().date(), "→", data.index.max().date())
print("Rows per regime attribution:")
print(data["regime"].value_counts().sort_index()
      .rename(lambda i: BY_INDEX[i]["name"]).to_string())

# %% [markdown]
# ## Scatterplots: WR index vs capacity factor
#
# For each of the 7 regimes and each of the 3 CF variables, scatter the
# daily-mean WR index (x) against the daily-mean capacity factor (y).
# The regression line shows the linear trend.

# %%
CF_VARS   = ["solar",   "wind_on",       "wind_off"]
CF_LABELS = ["Solar PV CF", "Wind onshore CF", "Wind offshore CF"]

fig, axes = plt.subplots(
    len(CF_VARS), len(WR_NAMES),
    figsize=(3.0 * len(WR_NAMES), 2.8 * len(CF_VARS)),
    sharex="col",
)

for col_i, regime in enumerate(WR_NAMES):
    color = BY_INDEX[col_i + 1]["color"]   # index 1..7

    for row_i, (cf_var, cf_label) in enumerate(zip(CF_VARS, CF_LABELS)):
        ax = axes[row_i, col_i]

        sub = data[[regime, cf_var]].dropna()
        x = sub[regime].values
        y = sub[cf_var].values

        ax.scatter(x, y, s=3, alpha=0.25, color=color, rasterized=True)

        # Regression line
        slope, intercept, r, p, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, intercept + slope * x_line, color="black",
                linewidth=1.2, zorder=5)

        ax.set_title(f"r={r:.2f}", fontsize=9, pad=2)
        ax.tick_params(labelsize=7)

        if col_i == 0:
            ax.set_ylabel(cf_label, fontsize=9)
        if row_i == len(CF_VARS) - 1:
            ax.set_xlabel(f"IWR ({regime})", fontsize=9)

        ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")

fig.suptitle("Daily WR index vs DE capacity factors (1979–2025)", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(paths.images_path / "10_wr_de_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/10_wr_de_scatter.png
# :name: fig-10-wr-de-scatter
# Scatterplots of daily-mean weather regime index (IWR, x-axis) against daily-mean
# capacity factor for Germany (y-axis) for all 7 regimes (columns) and three
# generation types (rows). The black line is the OLS regression; r is the Pearson
# correlation. Positive r means higher IWR → higher CF; negative r means the
# regime suppresses that generation type.
# ```

# %% [markdown]
# ## Boxplots: capacity factor by regime attribution
#
# Each day is attributed to one of the 7 regimes (lifecycle active) or
# "no regime" (index 0). Boxplots show the distribution of the daily
# capacity factor conditioned on that attribution.

# %%
REGIME_ORDER = list(range(8))
REGIME_LABELS = [BY_INDEX[i]["name"] for i in REGIME_ORDER]
REGIME_COLORS = [BY_INDEX[i]["color"] for i in REGIME_ORDER]

fig, axes = plt.subplots(1, len(CF_VARS), figsize=(14, 5), sharey=False)

for ax, cf_var, cf_label in zip(axes, CF_VARS, CF_LABELS):
    groups = [data.loc[data["regime"] == i, cf_var].dropna().values
              for i in REGIME_ORDER]

    bp = ax.boxplot(
        groups,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        flierprops=dict(marker=".", markersize=2, alpha=0.4, linestyle="none"),
        showfliers=True,
    )

    for patch, color in zip(bp["boxes"], REGIME_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, 9))
    ax.set_xticklabels(REGIME_LABELS, fontsize=10)
    ax.set_ylabel(cf_label, fontsize=11)
    ax.set_xlabel("Regime attribution", fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_title(cf_label, fontsize=11)
    ax.set_ylim(0, None)

fig.suptitle("DE capacity factor distributions by weather regime (daily, 1979–2025)",
             fontsize=12)
fig.tight_layout()
fig.savefig(paths.images_path / "10_wr_de_boxplot.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/10_wr_de_boxplot.png
# :name: fig-10-wr-de-boxplot
# Distributions of daily-mean capacity factors for Germany conditioned on
# lifecycle weather regime attribution. Each box spans the IQR; the horizontal
# line is the median; whiskers are 1.5×IQR. "no" = no active lifecycle regime.
# ```

# %% [markdown]
# ## Correlation matrix: solar, wind onshore, wind offshore
#
# How strongly are the three generation sources correlated in Germany?

# %%
corr_data = data[CF_VARS].rename(columns={
    "solar":    "Solar PV",
    "wind_on":  "Wind onshore",
    "wind_off": "Wind offshore",
}).dropna()

corr = corr_data.corr()
print("Pearson correlation matrix (daily, 1979–2025):")
print(corr.round(3))

n = len(corr)
fig, ax = plt.subplots(figsize=(5, 4))

im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(corr.columns, fontsize=11)
ax.set_yticklabels(corr.index, fontsize=11)

for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=12,
                color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")

ax.set_title("Capacity factor correlations — Germany (daily, 1979–2025)", fontsize=11)
fig.tight_layout()
fig.savefig(paths.images_path / "10_wr_de_corr.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/10_wr_de_corr.png
# :name: fig-10-wr-de-corr
# Pearson correlation matrix of daily-mean capacity factors for Germany.
# Wind onshore and offshore are strongly positively correlated; solar is
# negatively correlated with both wind variables (wind is stronger in winter,
# solar in summer).
# ```

# %% [markdown]
# ## Boxplots: meteorological variables by regime attribution
#
# Same regime conditioning as above, but for the three meteorological drivers:
# 2 m air temperature (°C), surface downwelling shortwave radiation (W m⁻²),
# and 10 m wind speed (m s⁻¹).

# %%
MET_VARS   = ["temp",       "rad",                               "wspeed"]
MET_LABELS = ["2 m temperature (°C)", "SW radiation (W m⁻²)", "10 m wind speed (m s⁻¹)"]

fig, axes = plt.subplots(1, len(MET_VARS), figsize=(14, 5), sharey=False)

for ax, met_var, met_label in zip(axes, MET_VARS, MET_LABELS):
    groups = [data.loc[data["regime"] == i, met_var].dropna().values
              for i in REGIME_ORDER]

    bp = ax.boxplot(
        groups,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        flierprops=dict(marker=".", markersize=2, alpha=0.4, linestyle="none"),
        showfliers=True,
    )

    for patch, color in zip(bp["boxes"], REGIME_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, 9))
    ax.set_xticklabels(REGIME_LABELS, fontsize=10)
    ax.set_ylabel(met_label, fontsize=11)
    ax.set_xlabel("Regime attribution", fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_title(met_label, fontsize=11)

fig.suptitle("DE meteorological variables by weather regime (daily, 1979–2025)",
             fontsize=12)
fig.tight_layout()
fig.savefig(paths.images_path / "10_wr_de_boxplot_met.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/10_wr_de_boxplot_met.png
# :name: fig-10-wr-de-boxplot-met
# Distributions of daily-mean meteorological variables for Germany conditioned on
# lifecycle weather regime attribution. Temperature and radiation show strong
# seasonal signals embedded in regime differences; wind speed reflects the
# circulation type directly.
# ```
