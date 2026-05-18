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
# # Lifecycle Inspection
#
# Two questions about the lifecycle attribution:
#
# 1. **How long does a regime hold?** — duration distributions for all 7 named
#    regimes and the no-regime state, shown as a jitter boxplot.
# 2. **What regime comes next?** — a transition matrix giving, for each current
#    state, the conditional probability of each next state when the current
#    period ends.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from wr.paths import ProjPaths
from wr.regimes import REGIMES, BY_INDEX, BY_NAME, WR_NAMES

paths = ProjPaths()

# %%
lc_info  = pd.read_csv(paths.lc_info_csv,        parse_dates=["onset", "decay"])
lc_no    = pd.read_csv(paths.lc_no_regime_csv,   parse_dates=["onset", "decay"])
lc_attr  = pd.read_csv(paths.lc_attribution_csv, parse_dates=["datetime"], index_col="datetime")

COLORS = {r["name"]: r["color"] for r in REGIMES}
RNG    = np.random.default_rng(42)

# %% [markdown]
# ## Figure 1 — Lifecycle duration jitter boxplot
#
# Duration is measured from lifecycle onset to decay.  The `lc_no_regime.csv`
# file stores durations in hours; `lc_info.csv` stores onset/decay timestamps
# from which we compute the duration in days.

# %%
# Named regimes: duration from lc_info
durations: dict[str, np.ndarray] = {}
for regime in WR_NAMES:
    subset = lc_info[lc_info["regime"] == regime]
    days   = (subset["decay"] - subset["onset"]).dt.total_seconds() / 86400
    durations[regime] = days.values

# No-regime: duration column is in hours
durations["no"] = lc_no["duration"].values / 24.0

# Ordered for plotting: no-regime first, then the 7 named regimes
PLOT_ORDER = ["no"] + WR_NAMES
n_groups   = len(PLOT_ORDER)

fig, ax = plt.subplots(figsize=(13, 5))

x_pos    = np.arange(n_groups)
box_data = [durations[r] for r in PLOT_ORDER]

bp = ax.boxplot(
    box_data,
    positions=x_pos,
    widths=0.5,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker=""),
    showfliers=False,
)

for patch, regime in zip(bp["boxes"], PLOT_ORDER):
    patch.set_facecolor(COLORS[regime])
    patch.set_alpha(0.35)

# Jitter points
for i, regime in enumerate(PLOT_ORDER):
    d      = durations[regime]
    jitter = RNG.uniform(-0.18, 0.18, len(d))
    ax.scatter(
        x_pos[i] + jitter, d,
        color=COLORS[regime], edgecolors="none",
        s=8, alpha=0.55, zorder=3,
    )

ax.set_xticks(x_pos)
ax.set_xticklabels(
    [f"{r}\n(n={len(durations[r])})" for r in PLOT_ORDER],
    fontsize=11,
)
ax.set_ylabel("Duration (days)", fontsize=12)
ax.set_title("Lifecycle duration by regime", fontsize=13)
ax.tick_params(axis="y", labelsize=11)
ax.set_ylim(bottom=0)

fig.tight_layout()
fig.savefig(paths.images_path / "08_lc_duration_jitter.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/08_lc_duration_jitter.png
# :name: fig-08-lc-duration-jitter
# Distribution of lifecycle durations (onset to decay) for the no-regime state
# and each of the 7 named weather regimes.  Boxes span the interquartile range;
# dots show individual lifecycles with horizontal jitter.
# ```

# %% [markdown]
# ## Figure 2 — Regime transition matrix
#
# We extract the sequence of distinct consecutive regime periods from the daily
# (12 UTC) lifecycle attribution time series, then count each (from → to)
# transition.  Counts are row-normalised to give conditional probabilities:
# P(next state = j | current state = i).

# %%
# Daily 12 UTC attribution
daily = lc_attr.loc[lc_attr.index.hour == 12, "lifecycle_wr_index"].dropna().astype(int)

# Sequence of distinct consecutive states (drop repeated same-state steps)
periods    = daily[daily != daily.shift(1)]
from_arr   = periods.values[:-1]
to_arr     = periods.values[1:]

# 8×8 count matrix (index 0 = no regime, 1-7 = named regimes)
N_STATES = 8
counts   = np.zeros((N_STATES, N_STATES), dtype=int)
for f, t in zip(from_arr, to_arr):
    counts[f, t] += 1

# Row-normalise → conditional probabilities
row_sums  = counts.sum(axis=1, keepdims=True)
trans_mat = np.where(row_sums > 0, counts / row_sums, np.nan)

state_labels = [BY_INDEX[i]["name"] for i in range(N_STATES)]
state_colors = [BY_INDEX[i]["color"] for i in range(N_STATES)]

# %%
fig, ax = plt.subplots(figsize=(9, 8))

# Heatmap (mask NaN cells)
im = ax.imshow(
    np.nan_to_num(trans_mat, nan=0.0),
    vmin=0, vmax=0.5,
    cmap="YlOrRd",
    aspect="equal",
)

# Annotate cells
for i in range(N_STATES):
    for j in range(N_STATES):
        val = trans_mat[i, j]
        if np.isnan(val):
            continue
        text_color = "white" if val > 0.35 else "black"
        ax.text(
            j, i, f"{val:.2f}",
            ha="center", va="center",
            fontsize=9, color=text_color, fontweight="bold",
        )

# Colour-coded axis tick labels
ax.set_xticks(range(N_STATES))
ax.set_yticks(range(N_STATES))
ax.set_xticklabels(state_labels, fontsize=11)
ax.set_yticklabels(state_labels, fontsize=11)

for tick, color in zip(ax.get_xticklabels(), state_colors):
    tick.set_color(color)
    tick.set_fontweight("bold")
for tick, color in zip(ax.get_yticklabels(), state_colors):
    tick.set_color(color)
    tick.set_fontweight("bold")

ax.set_xlabel("Next state", fontsize=12)
ax.set_ylabel("Current state", fontsize=12)
ax.set_title(
    "Regime transition matrix  (row-normalised, P(next | current))",
    fontsize=13,
)

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Transition probability", fontsize=11)

fig.tight_layout()
fig.savefig(paths.images_path / "08_transition_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/08_transition_matrix.png
# :name: fig-08-transition-matrix
# Regime transition matrix.  Each cell (i, j) shows the probability that the
# next distinct regime period is j given that the current period is i.  Rows
# sum to 1.  The sequence of distinct periods is derived from the daily (12 UTC)
# lifecycle attribution; consecutive timesteps with the same attribution are
# collapsed into one period before counting.
# ```

# %% [markdown]
# ## Figure 3 — Transition matrix including self-transitions
#
# Same daily (12 UTC) time series, but now every consecutive observation pair
# is counted — including staying in the same state.  High diagonal values
# reflect regime persistence; the off-diagonal structure shows preferred
# exit paths.

# %%
# All consecutive daily observation pairs (self-transitions included)
from_arr_all = daily.values[:-1]
to_arr_all   = daily.values[1:]

counts_all = np.zeros((N_STATES, N_STATES), dtype=int)
for f, t in zip(from_arr_all, to_arr_all):
    counts_all[f, t] += 1

row_sums_all  = counts_all.sum(axis=1, keepdims=True)
trans_mat_all = np.where(row_sums_all > 0, counts_all / row_sums_all, np.nan)

# %%
fig, ax = plt.subplots(figsize=(9, 8))

im = ax.imshow(
    np.nan_to_num(trans_mat_all, nan=0.0),
    vmin=0, vmax=1.0,
    cmap="YlOrRd",
    aspect="equal",
)

for i in range(N_STATES):
    for j in range(N_STATES):
        val = trans_mat_all[i, j]
        if np.isnan(val):
            continue
        text_color = "white" if val > 0.65 else "black"
        ax.text(
            j, i, f"{val:.2f}",
            ha="center", va="center",
            fontsize=9, color=text_color, fontweight="bold",
        )

ax.set_xticks(range(N_STATES))
ax.set_yticks(range(N_STATES))
ax.set_xticklabels(state_labels, fontsize=11)
ax.set_yticklabels(state_labels, fontsize=11)

for tick, color in zip(ax.get_xticklabels(), state_colors):
    tick.set_color(color)
    tick.set_fontweight("bold")
for tick, color in zip(ax.get_yticklabels(), state_colors):
    tick.set_color(color)
    tick.set_fontweight("bold")

ax.set_xlabel("Next state", fontsize=12)
ax.set_ylabel("Current state", fontsize=12)
ax.set_title(
    "Regime transition matrix — all observations  (P(t+1 | t), self-transitions included)",
    fontsize=13,
)

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Transition probability", fontsize=11)

fig.tight_layout()
fig.savefig(paths.images_path / "08_transition_matrix_all.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/08_transition_matrix_all.png
# :name: fig-08-transition-matrix-all
# Transition matrix counting every consecutive daily observation pair.  The
# diagonal captures day-to-day persistence; each row sums to 1.  Compare with
# the distinct-period matrix above: removing self-transitions redistributes
# the diagonal probability mass to reveal the preferred successor states.
# ```
