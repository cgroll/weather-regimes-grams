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
# # Weather Regime Maps — Mean CF by Country
#
# For each weather regime (and "no regime"), shows the mean capacity factor
# across all European countries as a choropleth map. Three generation types
# are shown side by side: solar PV, wind onshore, wind offshore.
#
# Data: PECD (hourly, 1979–2026) resampled to daily; Grams lifecycle
# attribution resampled to daily dominant regime.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import shapely.geometry as sg
from pathlib import Path

_CMAP_CF   = plt.get_cmap("PiYG")       # green = positive CF anomaly, pink = negative
_CMAP_TEMP = plt.get_cmap("coolwarm")   # red = warmer than average, blue = cooler

from wr.paths import ProjPaths
from wr.regimes import BY_INDEX, WR_NAMES

paths = ProjPaths()

PECD_PATH   = "/home/chris/research/world-of-energy/data/processed/pecd/pecd_regions.parquet"
NE_SHP      = "/home/chris/.local/share/cartopy/shapefiles/natural_earth/cultural/ne_10m_admin_0_countries.shp"

# European projection (ETRS89-LAEA)
PROJ_CRS = "EPSG:3035"

# PECD uses EL (Greece) and UK (Britain); naturalearth uses GR and GB
PECD_TO_ISO = {"EL": "GR", "UK": "GB"}

# %% [markdown]
# ## Load world shapefile

# %%
world_raw = gpd.read_file(NE_SHP)

# Build a single effective ISO2 column: prefer ISO_A2, fall back to ISO_A2_EH
world_raw["iso2"] = world_raw["ISO_A2"].where(
    world_raw["ISO_A2"] != "-99", world_raw["ISO_A2_EH"]
)

world = world_raw[["iso2", "NAME", "geometry"]].copy()
world = world.to_crs(PROJ_CRS)

# Clip world to a European geographic bounding box so the background plot never
# expands the axis limits beyond Europe (lon -27..50, lat 27..73 covers all PECD
# countries including Iceland, Turkey, and the Canaries).
eu_bbox = gpd.GeoDataFrame(
    geometry=[sg.box(-27, 27, 50, 73)], crs="EPSG:4326"
).to_crs(PROJ_CRS)
world_eu = gpd.clip(world, eu_bbox)

# %% [markdown]
# ## Load and align PECD + regime data

# %%
pecd = pd.read_parquet(PECD_PATH)

solar_h   = pecd["solar_photovoltaic_power_generation"]["capacity_factor_ratio"]
wind_on_h = pecd["wind_power_generation_onshore"]["capacity_factor_ratio"]
temp_h    = pecd["2m_air_temperature"]["value"] - 273.15   # K → °C

# Daily means — solar NaN (night) are skipped automatically
solar_d   = solar_h.resample("D").mean()
wind_on_d = wind_on_h.resample("D").mean()
temp_d    = temp_h.resample("D").mean()

# LC attribution: daily dominant regime (1950–2025, 3-hourly → daily mode)
lc = pd.read_csv(paths.lc_attribution_csv, parse_dates=["datetime"], index_col="datetime")
lc_daily = (
    lc["lifecycle_wr_index"]
    .resample("D")
    .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
    .astype(float)
)

# Restrict to overlap period (1979 onwards)
lc_daily = lc_daily.reindex(solar_d.index)

print("Overlap period:", solar_d.index.min().date(), "→", solar_d.index.max().date())
print("Days per regime:")
for i in range(8):
    n = (lc_daily == i).sum()
    print(f"  {BY_INDEX[i]['name']:>5}: {n:5d} days")

# %% [markdown]
# ## Compute per-regime anomalies (deviation from overall mean)
#
# For each country and variable, the anomaly for a regime is:
#   regime_mean_CF - overall_mean_CF
# Red = above average, blue = below average.

# %%
REGIME_IDS = list(range(8))

# Overall mean across all days (used as the baseline)
solar_clim   = solar_d.mean()
wind_on_clim = wind_on_d.mean()
temp_clim    = temp_d.mean()

def regime_anomalies(cf_daily: pd.DataFrame, clim: pd.Series,
                     lc: pd.Series) -> dict[int, pd.Series]:
    """Return {regime_idx: Series(country → mean anomaly)} for each regime."""
    out = {}
    for i in REGIME_IDS:
        mask = lc == i
        out[i] = cf_daily.loc[mask].mean() - clim
    return out

solar_means   = regime_anomalies(solar_d,   solar_clim,   lc_daily)
wind_on_means = regime_anomalies(wind_on_d, wind_on_clim, lc_daily)
temp_means    = regime_anomalies(temp_d,    temp_clim,    lc_daily)

# %% [markdown]
# ## Symmetric diverging colour scales (shared across all regimes)

# %%
def _abs_max(anom_dict):
    return pd.concat(anom_dict.values()).dropna().abs().quantile(0.97)

solar_lim   = _abs_max(solar_means)
wind_on_lim = _abs_max(wind_on_means)
temp_lim    = _abs_max(temp_means)

CMAPS = {
    "solar":   (_CMAP_CF,   -solar_lim,   solar_lim),
    "wind_on": (_CMAP_CF,   -wind_on_lim, wind_on_lim),
    "temp":    (_CMAP_TEMP, -temp_lim,    temp_lim),
}
CF_VARS  = list(CMAPS.keys())
CF_TITLES = {
    "solar":   "Solar PV CF anomaly",
    "wind_on": "Wind onshore CF anomaly",
    "temp":    "2 m temperature anomaly (°C)",
}

print("Colour scale limits (±):")
for k, (cmap, vmin, vmax) in CMAPS.items():
    print(f"  {k}: ±{vmax:.4f}")

# %% [markdown]
# ## Map extent and helper

# %%
# Hard-coded EPSG:3035 extent for mainland Europe
# (avoids Atlantic island territories like Azores/Canaries expanding the bbox)
XLIM = (2_600_000, 7_000_000)
YLIM = (1_400_000, 5_500_000)

all_iso_ne = [PECD_TO_ISO.get(c, c) for c in solar_d.columns]


def make_regime_map(regime_idx: int) -> plt.Figure:
    """One figure with 3 horizontal subplots for a single regime."""
    regime_name = BY_INDEX[regime_idx]["name"]
    regime_color = BY_INDEX[regime_idx]["color"]

    means_lookup = {
        "solar":   solar_means[regime_idx],
        "wind_on": wind_on_means[regime_idx],
        "temp":    temp_means[regime_idx],
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, cf_var in zip(axes, CF_VARS):
        cmap, vmin, vmax = CMAPS[cf_var]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Build per-country CF series → join to shapefile
        cf_series = means_lookup[cf_var].rename("cf")
        cf_series.index = [PECD_TO_ISO.get(c, c) for c in cf_series.index]
        cf_df = cf_series.reset_index().rename(columns={"index": "iso2"})

        geo = world.merge(cf_df, on="iso2", how="left")

        # Ocean background + European countries (grey background)
        # Use world_eu (clipped to Europe) so geopandas never expands axis limits
        # beyond the European bounding box.
        ax.set_facecolor("#d6eaf8")
        world_eu.plot(ax=ax, color="#cccccc", linewidth=0.3, edgecolor="white")

        # PECD countries coloured by CF
        has_data = geo[geo["cf"].notna()]
        no_data  = geo[geo["cf"].isna() & geo["iso2"].isin(all_iso_ne)]

        if len(has_data):
            has_data.plot(ax=ax, column="cf", cmap=cmap,
                          vmin=vmin, vmax=vmax,
                          linewidth=0.4, edgecolor="white")
        if len(no_data):
            # PECD country but variable not available (e.g. landlocked → no offshore)
            no_data.plot(ax=ax, color="#eeeeee", linewidth=0.4, edgecolor="white")

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, orientation="vertical")
        cbar.set_label("°C anomaly" if cf_var == "temp" else "CF anomaly", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        ax.set_axis_off()
        ax.set_title(CF_TITLES[cf_var], fontsize=12, pad=6)

    fig.suptitle(
        f"Regime: {regime_name} — CF anomaly vs overall mean, by country (1979–2025)",
        fontsize=14, fontweight="bold",
        color=regime_color if regime_name != "no" else "dimgrey",
        y=1.01,
    )
    fig.tight_layout()
    return fig


# %% [markdown]
# ## Generate one map per regime

# %%
for regime_idx in REGIME_IDS:
    name = BY_INDEX[regime_idx]["name"]
    fig = make_regime_map(regime_idx)
    out_path = paths.images_path / f"11_wr_maps_{name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")

# %% [markdown]
# ## Display maps

# %% [markdown]
# ### No regime
#
# ```{figure} ../../output/images/11_wr_maps_no.png
# :name: fig-11-wr-maps-no
# Mean capacity factors by country during days with **no active lifecycle regime**.
# ```

# %% [markdown]
# ### AT — Atlantic Trough
#
# ```{figure} ../../output/images/11_wr_maps_AT.png
# :name: fig-11-wr-maps-AT
# Mean capacity factors by country during **AT (Atlantic Trough)** regime days.
# ```

# %% [markdown]
# ### ZO — Zonal
#
# ```{figure} ../../output/images/11_wr_maps_ZO.png
# :name: fig-11-wr-maps-ZO
# Mean capacity factors by country during **ZO (Zonal)** regime days.
# ```

# %% [markdown]
# ### ScTr — Scandinavian Trough
#
# ```{figure} ../../output/images/11_wr_maps_ScTr.png
# :name: fig-11-wr-maps-ScTr
# Mean capacity factors by country during **ScTr (Scandinavian Trough)** regime days.
# ```

# %% [markdown]
# ### AR — Atlantic Ridge
#
# ```{figure} ../../output/images/11_wr_maps_AR.png
# :name: fig-11-wr-maps-AR
# Mean capacity factors by country during **AR (Atlantic Ridge)** regime days.
# ```

# %% [markdown]
# ### EuBL — European Blocking
#
# ```{figure} ../../output/images/11_wr_maps_EuBL.png
# :name: fig-11-wr-maps-EuBL
# Mean capacity factors by country during **EuBL (European Blocking)** regime days.
# ```

# %% [markdown]
# ### ScBL — Scandinavian Blocking
#
# ```{figure} ../../output/images/11_wr_maps_ScBL.png
# :name: fig-11-wr-maps-ScBL
# Mean capacity factors by country during **ScBL (Scandinavian Blocking)** regime days.
# ```

# %% [markdown]
# ### GL — Greenland Blocking
#
# ```{figure} ../../output/images/11_wr_maps_GL.png
# :name: fig-11-wr-maps-GL
# Mean capacity factors by country during **GL (Greenland Blocking)** regime days.
# ```
