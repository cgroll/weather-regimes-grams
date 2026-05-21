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
# # Low Wind Events and Weather Regimes — Germany
#
# Ranks all 2-day rolling windows of wind onshore CF in Germany from worst to
# best. For an equidistant grid of cut-offs N (2 000, 4 000, … up to the worst
# ~10 % of all 2-day windows ≈ 41 000), collects every hourly timestamp that
# contributed to any selected window and counts the associated lifecycle
# weather regime.
#
# Two questions are answered:
# 1. **Regime composition**: which regimes dominate during the worst low-wind
#    periods? (stacked bar)
# 2. **Information content**: how many distinct hours/days do the N worst
#    windows actually cover, and what fraction of the full dataset is that?
#    (line with dual y-axis)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from wr.paths import ProjPaths
from wr.regimes import BY_INDEX, WR_NAMES

paths = ProjPaths()

PECD_PATH = "/home/chris/research/world-of-energy/data/processed/pecd/pecd_regions.parquet"

# %% [markdown]
# ## Load data

# %%
pecd = pd.read_parquet(PECD_PATH)
de_wind_on_h = pecd[("wind_power_generation_onshore", "capacity_factor_ratio", "DE")]

# LC attribution: 3-hourly → forward-filled to hourly, aligned to PECD index
lc = pd.read_csv(paths.lc_attribution_csv, parse_dates=["datetime"], index_col="datetime")
lc_aligned = (
    lc["lifecycle_wr_index"]
    .resample("h").ffill()
    .reindex(de_wind_on_h.index, method="ffill")
    .fillna(0)
    .astype(np.int8)
)

TOTAL_HOURS = len(de_wind_on_h)
print(f"Total hourly observations : {TOTAL_HOURS:,}")
print(f"LC coverage               : {lc_aligned.index.min().date()} → {lc_aligned.index.max().date()}")

# %% [markdown]
# ## 2-day rolling window — sort worst first

# %%
rolling_2d  = de_wind_on_h.rolling(48, min_periods=48).mean().dropna()
sorted_ends = rolling_2d.sort_values().index        # ascending → worst first
N_WINDOWS   = len(sorted_ends)

print(f"Valid 2-day windows : {N_WINDOWS:,}")
print(f"10 % cut-off        : {int(N_WINDOWS * 0.1):,}")
print(f"Worst 3:")
print(rolling_2d.loc[sorted_ends[:3]].to_string())

# %% [markdown]
# ## Incremental regime accumulation (numpy, vectorised)

# %%
# Equidistant grid: step ≈ 2 000, from 2 000 up to ~10 % of all windows
N_MAX    = int(N_WINDOWS * 0.10)
N_STEP   = 2_000
N_VALUES = list(range(N_STEP, N_MAX + N_STEP, N_STEP))

# Pre-build timestamp → integer-position lookup (O(1) later)
ts_to_pos = {ts: i for i, ts in enumerate(de_wind_on_h.index)}
lc_arr    = lc_aligned.values          # numpy int8 array, shape (n_hours,)

seen_mask = np.zeros(TOTAL_HOURS, dtype=bool)
counts    = np.zeros(8, dtype=np.int64)
results   = []
ptr       = 0

for n_target in N_VALUES:
    while ptr < n_target:
        end_t   = sorted_ends[ptr]
        end_pos = ts_to_pos.get(end_t)
        if end_pos is not None:
            start_pos = max(0, end_pos - 47)
            positions = np.arange(start_pos, end_pos + 1)
            new_mask  = ~seen_mask[positions]
            if new_mask.any():
                new_pos = positions[new_mask]
                seen_mask[new_pos] = True
                counts += np.bincount(lc_arr[new_pos].astype(np.int64), minlength=8)
        ptr += 1

    total = counts.sum()
    results.append({
        "n":            n_target,
        "unique_hours": int(seen_mask.sum()),
        **{f"r{i}": counts[i] / total if total > 0 else 0.0 for i in range(8)},
    })
    if n_target % 10_000 == 0:
        print(f"  N={n_target:6,}  unique days={seen_mask.sum()/24:.0f}")

df = pd.DataFrame(results).set_index("n")
df["unique_days"] = df["unique_hours"] / 24
df["pct_hours"]   = df["unique_hours"] / TOTAL_HOURS * 100

print(df[["unique_days", "pct_hours"]].to_string())

# %% [markdown]
# ## Plot: regime composition + unique-day coverage

# %%
REGIME_NAMES  = [BY_INDEX[i]["name"]  for i in range(8)]
REGIME_COLORS = [BY_INDEX[i]["color"] for i in range(8)]

x         = np.arange(len(N_VALUES))
bar_width = 0.75

fig, (ax_bar, ax_line) = plt.subplots(
    2, 1, figsize=(16, 9),
    gridspec_kw={"height_ratios": [3, 1]},
)

# ── Stacked bar: regime fractions ────────────────────────────────────────────
bottoms = np.zeros(len(N_VALUES))
for i in range(8):
    vals = df[f"r{i}"].values
    ax_bar.bar(x, vals, bottom=bottoms, width=bar_width,
               color=REGIME_COLORS[i], label=REGIME_NAMES[i])
    bottoms += vals

ax_bar.set_xticks(x)
ax_bar.set_xticklabels([f"{n//1000}k" for n in N_VALUES], fontsize=9)
ax_bar.set_ylabel("Fraction of hours", fontsize=12)
ax_bar.set_ylim(0, 1)
ax_bar.set_title(
    "Weather regime composition of the N worst 2-day wind onshore periods — Germany\n"
    f"(N ranges over the worst 2–{N_MAX//1000}k windows, step 2k; "
    f"10 % of all {N_WINDOWS//1000}k 2-day windows)",
    fontsize=12,
)
ax_bar.legend(
    handles=[mpatches.Patch(color=REGIME_COLORS[i], label=REGIME_NAMES[i])
             for i in range(8)],
    loc="upper right", ncol=4, fontsize=10,
)
ax_bar.tick_params(axis="y", labelsize=10)
ax_bar.grid(axis="y", linewidth=0.5, alpha=0.4)

# ── Line: unique days / % of all hours (dual y-axis) ────────────────────────
ax_line.plot(x, df["unique_days"].values, color="steelblue", linewidth=2.0,
             marker="o", markersize=4)
ax_line.set_ylabel("Unique days covered", fontsize=11, color="steelblue")
ax_line.tick_params(axis="y", labelcolor="steelblue", labelsize=10)

ax_r = ax_line.twinx()
# Keep same scale: right axis just relabels left axis values as percentages
y_max = df["unique_days"].max() * 1.08
ax_line.set_ylim(0, y_max)
ax_r.set_ylim(0, y_max * 24 / TOTAL_HOURS * 100)
ax_r.set_ylabel("% of all hourly observations", fontsize=11, color="steelblue")
ax_r.tick_params(axis="y", labelcolor="steelblue", labelsize=10)

ax_line.set_xticks(x)
ax_line.set_xticklabels([f"{n//1000}k" for n in N_VALUES], fontsize=9)
ax_line.set_xlabel("N worst 2-day windows", fontsize=12)
ax_line.grid(axis="y", linewidth=0.5, alpha=0.4)

fig.tight_layout()
fig.savefig(paths.images_path / "13_low_wind_regime_stacks.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/13_low_wind_regime_stacks.png
# :name: fig-13-low-wind-regime-stacks
# Top: stacked bar chart of the regime composition for the N worst 2-day wind
# onshore windows in Germany. Bottom: unique days (left axis) and fraction of
# all hourly observations (right axis) covered by those windows. The slow
# growth of unique days reveals strong temporal clustering of extreme low-wind
# events.
# ```

# %% [markdown]
# ## Predictive regime analysis — lagged attribution
#
# Same window selection, but instead of looking up the regime *during* the
# low-wind event, we look up the regime **LAG days before** each selected
# window. This reveals which circulation patterns systematically precede
# extreme low-wind periods.

# %%
def compute_lag_fracs(lag_days: int, sorted_ends_input=None) -> pd.DataFrame:
    """
    For each N in N_VALUES, collect the hourly regime observations that
    occurred LAG_DAYS before each of the N worst 2-day windows.
    sorted_ends_input defaults to the wind onshore sorted_ends.
    Returns a DataFrame indexed by N with columns r0..r7 (fractions) and
    unique_hours / unique_days.
    """
    if sorted_ends_input is None:
        sorted_ends_input = sorted_ends
    lag_h     = lag_days * 24
    seen_mask = np.zeros(TOTAL_HOURS, dtype=bool)
    counts    = np.zeros(8, dtype=np.int64)
    results   = []
    ptr       = 0

    for n_target in N_VALUES:
        while ptr < n_target:
            end_t   = sorted_ends_input[ptr]
            end_pos = ts_to_pos.get(end_t)
            if end_pos is not None:
                lag_end   = end_pos - lag_h          # end of the lagged window
                lag_start = lag_end - 47             # start of the lagged window (48 h)
                if lag_end >= 0 and lag_start < TOTAL_HOURS:
                    lag_start = max(0, lag_start)
                    positions = np.arange(lag_start, lag_end + 1)
                    new_mask  = ~seen_mask[positions]
                    if new_mask.any():
                        new_pos = positions[new_mask]
                        seen_mask[new_pos] = True
                        counts += np.bincount(lc_arr[new_pos].astype(np.int64), minlength=8)
            ptr += 1

        total = counts.sum()
        results.append({
            "n":            n_target,
            "unique_hours": int(seen_mask.sum()),
            **{f"r{i}": counts[i] / total if total > 0 else 0.0 for i in range(8)},
        })

    out = pd.DataFrame(results).set_index("n")
    out["unique_days"] = out["unique_hours"] / 24
    out["pct_hours"]   = out["unique_hours"] / TOTAL_HOURS * 100
    return out


LAG_DAYS = [5, 10, 15, 20, 25]

lag_dfs = {}
for lag in LAG_DAYS:
    print(f"Computing lag = {lag} days …", end=" ", flush=True)
    lag_dfs[lag] = compute_lag_fracs(lag)
    print(f"done  (N={N_VALUES[-1]}: {lag_dfs[lag]['unique_days'].iloc[-1]:.0f} unique days)")

# %% [markdown]
# ## Plot: one figure per lag

# %%
def plot_lag_chart(df_lag: pd.DataFrame, lag_days: int) -> plt.Figure:
    fig, (ax_bar, ax_line) = plt.subplots(
        2, 1, figsize=(16, 9),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # ── Stacked bar ──────────────────────────────────────────────────────────
    bottoms = np.zeros(len(N_VALUES))
    for i in range(8):
        vals = df_lag[f"r{i}"].values
        ax_bar.bar(x, vals, bottom=bottoms, width=bar_width,
                   color=REGIME_COLORS[i], label=REGIME_NAMES[i])
        bottoms += vals

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"{n//1000}k" for n in N_VALUES], fontsize=9)
    ax_bar.set_ylabel("Fraction of hours", fontsize=12)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_title(
        f"Regime composition {lag_days} days BEFORE the N worst 2-day wind onshore events — Germany\n"
        f"(worst 2k–{N_MAX//1000}k windows, step 2k)",
        fontsize=12,
    )
    ax_bar.legend(
        handles=[mpatches.Patch(color=REGIME_COLORS[i], label=REGIME_NAMES[i])
                 for i in range(8)],
        loc="upper right", ncol=4, fontsize=10,
    )
    ax_bar.tick_params(axis="y", labelsize=10)
    ax_bar.grid(axis="y", linewidth=0.5, alpha=0.4)

    # ── Unique lagged days + % ────────────────────────────────────────────────
    ax_line.plot(x, df_lag["unique_days"].values, color="steelblue",
                 linewidth=2.0, marker="o", markersize=4)
    ax_line.set_ylabel("Unique lagged\ndays covered", fontsize=11, color="steelblue")
    ax_line.tick_params(axis="y", labelcolor="steelblue", labelsize=10)

    ax_r = ax_line.twinx()
    y_max = df_lag["unique_days"].max() * 1.08
    ax_line.set_ylim(0, y_max)
    ax_r.set_ylim(0, y_max * 24 / TOTAL_HOURS * 100)
    ax_r.set_ylabel("% of all hourly\nobservations", fontsize=11, color="steelblue")
    ax_r.tick_params(axis="y", labelcolor="steelblue", labelsize=10)

    ax_line.set_xticks(x)
    ax_line.set_xticklabels([f"{n//1000}k" for n in N_VALUES], fontsize=9)
    ax_line.set_xlabel("N worst 2-day windows", fontsize=12)
    ax_line.grid(axis="y", linewidth=0.5, alpha=0.4)

    fig.tight_layout()
    return fig


for lag in LAG_DAYS:
    fig = plot_lag_chart(lag_dfs[lag], lag)
    fname = f"13_low_wind_lag_{lag:02d}d.png"
    fig.savefig(paths.images_path / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

# Also wire the simultaneous wind chart through the generalised helper so the
# solar section can reuse it without re-running the accumulation
plot_simul_chart = None   # defined in solar section below; wind chart already saved

# %% [markdown]
# ### Lag 5 days
#
# ```{figure} ../../output/images/13_low_wind_lag_05d.png
# :name: fig-13-low-wind-lag-05d
# Regime composition 5 days before the N worst 2-day wind onshore events.
# ```

# %% [markdown]
# ### Lag 10 days
#
# ```{figure} ../../output/images/13_low_wind_lag_10d.png
# :name: fig-13-low-wind-lag-10d
# Regime composition 10 days before the N worst 2-day wind onshore events.
# ```

# %% [markdown]
# ### Lag 15 days
#
# ```{figure} ../../output/images/13_low_wind_lag_15d.png
# :name: fig-13-low-wind-lag-15d
# Regime composition 15 days before the N worst 2-day wind onshore events.
# ```

# %% [markdown]
# ### Lag 20 days
#
# ```{figure} ../../output/images/13_low_wind_lag_20d.png
# :name: fig-13-low-wind-lag-20d
# Regime composition 20 days before the N worst 2-day wind onshore events.
# ```

# %% [markdown]
# ### Lag 25 days
#
# ```{figure} ../../output/images/13_low_wind_lag_25d.png
# :name: fig-13-low-wind-lag-25d
# Regime composition 25 days before the N worst 2-day wind onshore events.
# ```

# %% [markdown]
# ---
# ## Solar PV CF — same analysis
#
# 2-day rolling window on hourly solar CF; nighttime NaN excluded from mean
# via `min_periods=1` so the rolling value reflects daytime capacity factor.

# %%
de_solar_h = pecd[("solar_photovoltaic_power_generation", "capacity_factor_ratio", "DE")]

rolling_solar   = de_solar_h.rolling(48, min_periods=1).mean().dropna()
sorted_ends_sol = rolling_solar.sort_values().index      # worst first
print(f"Solar valid 2-day windows: {len(sorted_ends_sol):,}")
print(f"Worst 3 (mean daytime CF):")
print(rolling_solar.loc[sorted_ends_sol[:3]].to_string())

# %% [markdown]
# ### Simultaneous regime composition — solar

# %%
seen_mask_sol = np.zeros(TOTAL_HOURS, dtype=bool)
counts_sol    = np.zeros(8, dtype=np.int64)
results_sol   = []
ptr_sol       = 0

for n_target in N_VALUES:
    while ptr_sol < n_target:
        end_t   = sorted_ends_sol[ptr_sol]
        end_pos = ts_to_pos.get(end_t)
        if end_pos is not None:
            start_pos = max(0, end_pos - 47)
            positions = np.arange(start_pos, end_pos + 1)
            new_mask  = ~seen_mask_sol[positions]
            if new_mask.any():
                new_pos = positions[new_mask]
                seen_mask_sol[new_pos] = True
                counts_sol += np.bincount(lc_arr[new_pos].astype(np.int64), minlength=8)
        ptr_sol += 1

    total = counts_sol.sum()
    results_sol.append({
        "n":            n_target,
        "unique_hours": int(seen_mask_sol.sum()),
        **{f"r{i}": counts_sol[i] / total if total > 0 else 0.0 for i in range(8)},
    })

df_sol = pd.DataFrame(results_sol).set_index("n")
df_sol["unique_days"] = df_sol["unique_hours"] / 24
df_sol["pct_hours"]   = df_sol["unique_hours"] / TOTAL_HOURS * 100

# ── Plot simultaneous solar ──────────────────────────────────────────────────
def plot_simul_chart(df_in: pd.DataFrame, var_label: str, fname: str) -> None:
    fig, (ax_bar, ax_line) = plt.subplots(
        2, 1, figsize=(16, 9),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    bottoms = np.zeros(len(N_VALUES))
    for i in range(8):
        vals = df_in[f"r{i}"].values
        ax_bar.bar(x, vals, bottom=bottoms, width=bar_width,
                   color=REGIME_COLORS[i], label=REGIME_NAMES[i])
        bottoms += vals
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"{n//1000}k" for n in N_VALUES], fontsize=9)
    ax_bar.set_ylabel("Fraction of hours", fontsize=12)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_title(
        f"Weather regime composition of the N worst 2-day {var_label} periods — Germany\n"
        f"(worst 2k–{N_MAX//1000}k windows, step 2k)",
        fontsize=12,
    )
    ax_bar.legend(
        handles=[mpatches.Patch(color=REGIME_COLORS[i], label=REGIME_NAMES[i])
                 for i in range(8)],
        loc="upper right", ncol=4, fontsize=10,
    )
    ax_bar.tick_params(axis="y", labelsize=10)
    ax_bar.grid(axis="y", linewidth=0.5, alpha=0.4)

    ax_line.plot(x, df_in["unique_days"].values, color="steelblue",
                 linewidth=2.0, marker="o", markersize=4)
    ax_line.set_ylabel("Unique days\ncovered", fontsize=11, color="steelblue")
    ax_line.tick_params(axis="y", labelcolor="steelblue", labelsize=10)
    ax_r = ax_line.twinx()
    y_max = df_in["unique_days"].max() * 1.08
    ax_line.set_ylim(0, y_max)
    ax_r.set_ylim(0, y_max * 24 / TOTAL_HOURS * 100)
    ax_r.set_ylabel("% of all hourly\nobservations", fontsize=11, color="steelblue")
    ax_r.tick_params(axis="y", labelcolor="steelblue", labelsize=10)
    ax_line.set_xticks(x)
    ax_line.set_xticklabels([f"{n//1000}k" for n in N_VALUES], fontsize=9)
    ax_line.set_xlabel("N worst 2-day windows", fontsize=12)
    ax_line.grid(axis="y", linewidth=0.5, alpha=0.4)

    fig.tight_layout()
    fig.savefig(paths.images_path / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

plot_simul_chart(df_sol, "Solar PV CF", "13_solar_regime_stacks.png")

# %% [markdown]
# ```{figure} ../../output/images/13_solar_regime_stacks.png
# :name: fig-13-solar-regime-stacks
# Simultaneous regime composition for the N worst 2-day solar PV CF periods.
# ```

# %% [markdown]
# ### Lagged regime composition — solar

# %%
def plot_lag_chart_labeled(df_lag: pd.DataFrame, lag_days: int,
                           var_label: str, fname: str) -> None:
    fig, (ax_bar, ax_line) = plt.subplots(
        2, 1, figsize=(16, 9),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    bottoms = np.zeros(len(N_VALUES))
    for i in range(8):
        vals = df_lag[f"r{i}"].values
        ax_bar.bar(x, vals, bottom=bottoms, width=bar_width,
                   color=REGIME_COLORS[i], label=REGIME_NAMES[i])
        bottoms += vals
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"{n//1000}k" for n in N_VALUES], fontsize=9)
    ax_bar.set_ylabel("Fraction of hours", fontsize=12)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_title(
        f"Regime composition {lag_days} days BEFORE the N worst 2-day {var_label} events — Germany\n"
        f"(worst 2k–{N_MAX//1000}k windows, step 2k)",
        fontsize=12,
    )
    ax_bar.legend(
        handles=[mpatches.Patch(color=REGIME_COLORS[i], label=REGIME_NAMES[i])
                 for i in range(8)],
        loc="upper right", ncol=4, fontsize=10,
    )
    ax_bar.tick_params(axis="y", labelsize=10)
    ax_bar.grid(axis="y", linewidth=0.5, alpha=0.4)

    ax_line.plot(x, df_lag["unique_days"].values, color="steelblue",
                 linewidth=2.0, marker="o", markersize=4)
    ax_line.set_ylabel("Unique lagged\ndays covered", fontsize=11, color="steelblue")
    ax_line.tick_params(axis="y", labelcolor="steelblue", labelsize=10)
    ax_r = ax_line.twinx()
    y_max = df_lag["unique_days"].max() * 1.08
    ax_line.set_ylim(0, y_max)
    ax_r.set_ylim(0, y_max * 24 / TOTAL_HOURS * 100)
    ax_r.set_ylabel("% of all hourly\nobservations", fontsize=11, color="steelblue")
    ax_r.tick_params(axis="y", labelcolor="steelblue", labelsize=10)
    ax_line.set_xticks(x)
    ax_line.set_xticklabels([f"{n//1000}k" for n in N_VALUES], fontsize=9)
    ax_line.set_xlabel("N worst 2-day windows", fontsize=12)
    ax_line.grid(axis="y", linewidth=0.5, alpha=0.4)

    fig.tight_layout()
    fig.savefig(paths.images_path / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


solar_lag_dfs = {}
for lag in LAG_DAYS:
    print(f"Solar lag = {lag} days …", end=" ", flush=True)
    solar_lag_dfs[lag] = compute_lag_fracs(lag, sorted_ends_input=sorted_ends_sol)
    print(f"done  (N={N_VALUES[-1]}: {solar_lag_dfs[lag]['unique_days'].iloc[-1]:.0f} unique days)")

for lag in LAG_DAYS:
    fname = f"13_solar_lag_{lag:02d}d.png"
    plot_lag_chart_labeled(solar_lag_dfs[lag], lag, "Solar PV CF", fname)

# %% [markdown]
# ### Lag 5 days — solar
#
# ```{figure} ../../output/images/13_solar_lag_05d.png
# :name: fig-13-solar-lag-05d
# Regime composition 5 days before the N worst 2-day solar CF events.
# ```

# %% [markdown]
# ### Lag 10 days — solar
#
# ```{figure} ../../output/images/13_solar_lag_10d.png
# :name: fig-13-solar-lag-10d
# Regime composition 10 days before the N worst 2-day solar CF events.
# ```

# %% [markdown]
# ### Lag 15 days — solar
#
# ```{figure} ../../output/images/13_solar_lag_15d.png
# :name: fig-13-solar-lag-15d
# Regime composition 15 days before the N worst 2-day solar CF events.
# ```

# %% [markdown]
# ### Lag 20 days — solar
#
# ```{figure} ../../output/images/13_solar_lag_20d.png
# :name: fig-13-solar-lag-20d
# Regime composition 20 days before the N worst 2-day solar CF events.
# ```

# %% [markdown]
# ### Lag 25 days — solar
#
# ```{figure} ../../output/images/13_solar_lag_25d.png
# :name: fig-13-solar-lag-25d
# Regime composition 25 days before the N worst 2-day solar CF events.
# ```
