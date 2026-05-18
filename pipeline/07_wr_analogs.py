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
# # Weather Regime Analog Analysis
#
# For each of the 7 weather regimes we find the date that achieved the highest
# regime index (IWR) value in the full record.  We then search the archive for
# the 9 most similar states — measured by Euclidean distance in the
# 7-dimensional WRI vector space — subject to the constraint that every analog
# must originate from a **different lifecycle** than the query date and all
# previously selected analogs.  This prevents redundant picks of temporally
# adjacent timesteps that belong to the same weather event.
#
# ## Plots produced
#
# 1. **Jitter boxplot** — initial distances of the 50 analogs to the query state,
#    one group per regime.
# 2. **30-day divergence** — how Euclidean distance between each analog pair
#    evolves when both trajectories are propagated forward in parallel.
# 3. **Global pairwise distance histogram** — distribution of all pairwise
#    WRI distances (subsampled) as a reference background.
# 4. **Per-regime pairwise distances** — same histogram restricted to timesteps
#    attributed to each regime, compared with the global background.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import pdist

from wr.paths import ProjPaths
from wr.regimes import REGIMES, BY_INDEX, BY_NAME, WR_NAMES

paths = ProjPaths()

# %%
wri     = pd.read_csv(paths.wri_csv,            parse_dates=["datetime"], index_col="datetime")
lc_info = pd.read_csv(paths.lc_info_csv,        parse_dates=["onset", "decay"])
lc_attr = pd.read_csv(paths.lc_attribution_csv, parse_dates=["datetime"], index_col="datetime")

WRI_ARR = wri[WR_NAMES].values   # (N, 7) — raw array for fast distance computation
N       = len(wri)

COLORS = {r["name"]: r["color"] for r in REGIMES}

# %%
N_ANALOGS    = 50
HORIZON_DAYS = 30
RNG          = np.random.default_rng(42)

# %% [markdown]
# ## Lifecycle membership
#
# Each timestep is tagged with the set of lifecycle IDs it belongs to.  A
# lifecycle ID is `"<regime>_<number>"` for active-regime periods and
# `"no_<group>"` for contiguous no-regime intervals.  Two timesteps from
# different lifecycle sets are considered **independent** for the purpose of
# analog selection.

# %%
# Build (onset, decay, lc_id) tuples for fast membership checks
lc_intervals = [
    (row["onset"], row["decay"], f"{row['regime']}_{row['number']}")
    for _, row in lc_info.iterrows()
]

# Tag contiguous no-regime periods
no_mask   = lc_attr["lifecycle_wr_index"].reindex(wri.index, fill_value=0) == 0
changes   = no_mask != no_mask.shift(1, fill_value=False)
no_groups = changes.cumsum()
no_ids    = no_groups.where(no_mask)  # NaN for regime periods
no_id_map = {ts: f"no_{int(gid)}" for ts, gid in no_ids.items() if not pd.isna(gid)}


def lifecycle_ids(ts: pd.Timestamp) -> frozenset:
    """Return all lifecycle IDs that contain *ts*."""
    ids = {lc_id for onset, decay, lc_id in lc_intervals if onset <= ts < decay}
    if ts in no_id_map:
        ids.add(no_id_map[ts])
    return frozenset(ids)


# %% [markdown]
# ## Analog selection
#
# For a given query timestamp, we rank all other timesteps by Euclidean distance
# in WRI space, then greedily accept candidates whose lifecycle set does not
# overlap with any already-selected lifecycle (including the query's).

# %%
def find_analogs(
    query_ts: pd.Timestamp,
    n_analogs: int = N_ANALOGS,
    min_future_days: int = HORIZON_DAYS,
) -> tuple[list[pd.Timestamp], list[float]]:
    """Return (analog_timestamps, initial_distances) of length n_analogs."""
    q_idx   = wri.index.get_loc(query_ts)
    q_vec   = WRI_ARR[q_idx]
    dists   = np.sqrt(((WRI_ARR - q_vec) ** 2).sum(axis=1))
    order   = np.argsort(dists)

    # Latest index that still leaves min_future_days of data
    horizon_steps  = min_future_days * 8   # 8 × 3 h = 24 h
    last_valid_idx = N - horizon_steps - 1

    q_lcs    = lifecycle_ids(query_ts)
    used_lcs = set(q_lcs)
    sel_ts, sel_d = [], []

    for idx in order:
        if idx == q_idx or idx > last_valid_idx:
            continue
        ts   = wri.index[idx]
        lcs  = lifecycle_ids(ts)
        if lcs.isdisjoint(used_lcs):
            sel_ts.append(ts)
            sel_d.append(float(dists[idx]))
            used_lcs.update(lcs)
            if len(sel_ts) == n_analogs:
                break

    return sel_ts, sel_d


# %% [markdown]
# ## Query dates and analog selection for all 7 regimes

# %%
# Restrict query candidates to timesteps that have HORIZON_DAYS of future data
horizon_steps   = HORIZON_DAYS * 8
last_valid_ts   = wri.index[-(horizon_steps + 1)]

analog_results: dict = {}
for regime in WR_NAMES:
    query_ts   = wri.loc[wri.index <= last_valid_ts, regime].idxmax()
    query_iwr  = wri.loc[query_ts, regime]
    a_ts, a_d  = find_analogs(query_ts)
    analog_results[regime] = {
        "query_ts":    query_ts,
        "query_iwr":   query_iwr,
        "analog_ts":   a_ts,
        "analog_dist": np.array(a_d),
    }
    print(
        f"{regime:5s}  query={query_ts.date()}  IWR={query_iwr:.2f}  "
        f"analog distances: {np.array(a_d).round(2)}"
    )

# %% [markdown]
# ## 30-day divergence trajectories
#
# For each analog pair `(query, analog_i)`, compute the Euclidean distance at
# daily intervals when both time series are shifted forward in lockstep:
# `dist(WRI[query + Δt], WRI[analog_i + Δt])` for Δt = 0, 1, …, 30 days.

# %%
day_offsets = np.arange(HORIZON_DAYS + 1)   # 0 … 30

for regime, res in analog_results.items():
    query_ts = res["query_ts"]
    traj     = np.full((len(day_offsets), N_ANALOGS), np.nan)

    for k, day in enumerate(day_offsets):
        delta    = pd.Timedelta(days=int(day))
        q_future = query_ts + delta
        if q_future not in wri.index:
            break
        q_vec = wri.loc[q_future, WR_NAMES].values

        for j, a_ts in enumerate(res["analog_ts"]):
            a_future = a_ts + delta
            if a_future not in wri.index:
                continue
            a_vec      = wri.loc[a_future, WR_NAMES].values
            traj[k, j] = np.sqrt(np.sum((q_vec - a_vec) ** 2))

    res["traj_dists"] = traj   # shape (31, 9)

# %% [markdown]
# ## Figure 1 — Jitter boxplot of initial analog distances

# %%
fig, ax = plt.subplots(figsize=(12, 5))

x_positions = np.arange(len(WR_NAMES))
box_data    = [analog_results[r]["analog_dist"] for r in WR_NAMES]

bp = ax.boxplot(
    box_data,
    positions=x_positions,
    widths=0.5,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker=""),   # hide default fliers; we draw jitter instead
)

for i, (patch, regime) in enumerate(zip(bp["boxes"], WR_NAMES)):
    patch.set_facecolor(COLORS[regime])
    patch.set_alpha(0.35)

# Jittered individual points
for i, (regime, res) in enumerate(analog_results.items()):
    jitter = RNG.uniform(-0.15, 0.15, N_ANALOGS)
    ax.scatter(
        x_positions[i] + jitter,
        res["analog_dist"],
        color=COLORS[regime],
        edgecolors="black",
        linewidths=0.5,
        s=55,
        zorder=3,
    )

ax.set_xticks(x_positions)
ax.set_xticklabels(
    [f"{r}\n{analog_results[r]['query_ts'].strftime('%Y-%m-%d')}" for r in WR_NAMES],
    fontsize=11,
)
ax.set_ylabel("Euclidean distance in WRI space", fontsize=12)
ax.set_title(
    f"Initial WRI distance: {N_ANALOGS} analogs per regime (query = max-IWR date)",
    fontsize=13,
)
ax.tick_params(axis="y", labelsize=11)
fig.tight_layout()
fig.savefig(paths.images_path / "07_analog_jitter.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/07_analog_jitter.png
# :name: fig-07-analog-jitter
# Euclidean distance in 7-dimensional WRI space between each regime's peak-IWR
# date and its 50 closest analogs from independent lifecycles.  Boxes span the
# interquartile range; individual points are shown with horizontal jitter.
# ```

# %% [markdown]
# ## Figure 2 — 30-day divergence of analog pairs

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
axes_flat = axes.flatten()

for i, regime in enumerate(WR_NAMES):
    ax  = axes_flat[i]
    res = analog_results[regime]
    traj = res["traj_dists"]   # (31, 9)

    for j in range(N_ANALOGS):
        ax.plot(
            day_offsets, traj[:, j],
            color=COLORS[regime], alpha=0.45, linewidth=1.4,
        )

    # Median trajectory
    med = np.nanmedian(traj, axis=1)
    ax.plot(day_offsets, med, color=COLORS[regime], linewidth=2.5, label="median")

    # Mark the initial distance (day 0)
    ax.scatter(
        np.zeros(N_ANALOGS), traj[0, :],
        color=COLORS[regime], edgecolors="black", linewidths=0.5, s=40, zorder=4,
    )

    ax.set_title(
        f"{regime}  (query {res['query_ts'].strftime('%Y-%m-%d')})",
        fontsize=11,
    )
    ax.set_xlabel("Days forward", fontsize=10)
    ax.set_ylabel("Euclidean distance", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.set_xlim(0, HORIZON_DAYS)
    ax.set_ylim(bottom=0)

# Hide unused 8th panel
axes_flat[-1].set_visible(False)

fig.suptitle(
    "WRI state divergence: analog pairs tracked 30 days forward",
    fontsize=13, y=1.01,
)
fig.tight_layout()
fig.savefig(paths.images_path / "07_analog_divergence.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/07_analog_divergence.png
# :name: fig-07-analog-divergence
# Euclidean distance between each regime's peak-IWR state and its 50 analogs,
# tracked day-by-day for 30 days.  Both the query trajectory and each analog
# trajectory are advanced forward in lockstep.  Thin lines = individual analogs;
# thick line = median.
# ```

# %% [markdown]
# ## Figure 3 — Global pairwise distance distribution
#
# We subsample the full time series and compute all pairwise Euclidean distances
# as a reference background for the regime-specific distributions below.

# %%
SUBSAMPLE_N = 5000
idx_sample  = RNG.choice(N, size=SUBSAMPLE_N, replace=False)
all_dists   = pdist(WRI_ARR[np.sort(idx_sample)])

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(all_dists, bins=80, color="steelblue", alpha=0.8, edgecolor="none", density=True)
ax.set_xlabel("Euclidean distance in WRI space", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title(
    f"All pairwise WRI distances  (random subsample n={SUBSAMPLE_N:,})",
    fontsize=13,
)
ax.tick_params(labelsize=11)
fig.tight_layout()
fig.savefig(paths.images_path / "07_pairwise_dist.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/07_pairwise_dist.png
# :name: fig-07-pairwise-dist
# Distribution of all pairwise Euclidean distances in the 7-dimensional WRI
# vector space, based on a random subsample of 5,000 timesteps.
# ```

# %% [markdown]
# ## Figure 4 — Per-regime pairwise distance distributions
#
# For each regime (and the no-regime state), we subsample timesteps that carry
# that lifecycle attribution and compute their pairwise WRI distances, then
# overlay the global background for comparison.

# %%
REGIME_SUBSAMPLE = 3000

fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
axes_flat = axes.flatten()

# Global background kernel (use the already-computed all_dists)
bins = np.linspace(0, all_dists.max() * 1.05, 80)

for i, regime_idx in enumerate(range(8)):   # 0 = no regime, 1-7 = named regimes
    ax         = axes_flat[i]
    regime_name = BY_INDEX[regime_idx]["name"]
    color       = BY_INDEX[regime_idx]["color"]

    # Timesteps attributed to this regime
    mask  = lc_attr["lifecycle_wr_index"].reindex(wri.index, fill_value=0) == regime_idx
    ts_in = wri.index[mask]
    n_in  = len(ts_in)

    ax.hist(
        all_dists, bins=bins, color="lightgrey", alpha=0.8,
        edgecolor="none", density=True, label="all",
    )

    if n_in >= 2:
        k        = min(REGIME_SUBSAMPLE, n_in)
        sample   = RNG.choice(n_in, size=k, replace=False)
        sub_arr  = wri.loc[ts_in[sample], WR_NAMES].values
        r_dists  = pdist(sub_arr)
        ax.hist(
            r_dists, bins=bins, color=color, alpha=0.7,
            edgecolor="none", density=True, label=f"{regime_name} (n={n_in:,})",
        )

    long_name = BY_INDEX[regime_idx]["long_name"]
    ax.set_title(f"{regime_name} — {long_name}", fontsize=10)
    ax.set_xlabel("Euclidean distance", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)

fig.suptitle(
    "Pairwise WRI distances: within-regime vs. global background",
    fontsize=13, y=1.01,
)
fig.tight_layout()
fig.savefig(paths.images_path / "07_pairwise_dist_regimes.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/07_pairwise_dist_regimes.png
# :name: fig-07-pairwise-dist-regimes
# Per-regime pairwise WRI distance distributions (coloured) overlaid on the
# global background (grey).  A distribution shifted to the left indicates that
# timesteps carrying that regime label are more clustered in WRI space than
# random pairs, confirming that the lifecycle attribution captures coherent
# atmospheric states.
# ```

# %% [markdown]
# ## Figure 5 — Regime attribution heatmap across analogs and time
#
# For each analog and each future day we look up the lifecycle-attributed regime
# (`lifecycle_wr_index`) and colour-code it.  One row per analog, one column per
# day; the query date's own trajectory is shown as the top row.

# %%
# Build regime-attribution arrays: shape (N_ANALOGS+1, HORIZON_DAYS+1)
# Row 0 = query trajectory; rows 1..N_ANALOGS = analog trajectories.

lc_attr_idx = lc_attr["lifecycle_wr_index"].reindex(wri.index, fill_value=0)

def regime_trajectory(start_ts: pd.Timestamp) -> np.ndarray:
    """Return array of lifecycle_wr_index values at day 0, 1, …, HORIZON_DAYS."""
    out = np.full(len(day_offsets), -1, dtype=int)
    for k, day in enumerate(day_offsets):
        ts = start_ts + pd.Timedelta(days=int(day))
        if ts in lc_attr_idx.index:
            out[k] = int(lc_attr_idx.loc[ts])
    return out


def regime_trajectory_back(start_ts: pd.Timestamp) -> np.ndarray:
    """Return array of lifecycle_wr_index values at day 0, -1, …, -HORIZON_DAYS.

    Index k holds the attribution k days *before* start_ts.
    """
    out = np.full(len(day_offsets), -1, dtype=int)
    for k, day in enumerate(day_offsets):
        ts = start_ts - pd.Timedelta(days=int(day))
        if ts in lc_attr_idx.index:
            out[k] = int(lc_attr_idx.loc[ts])
    return out


fig, axes = plt.subplots(
    7, 1, figsize=(14, 14),
    gridspec_kw={"hspace": 0.35},
)

for ax, regime in zip(axes, WR_NAMES):
    res = analog_results[regime]

    # Build (1 + N_ANALOGS) × (HORIZON_DAYS+1) attribution matrix
    rows = [regime_trajectory(res["query_ts"])]
    for a_ts in res["analog_ts"]:
        rows.append(regime_trajectory(a_ts))
    attr_mat = np.array(rows)   # shape (51, 31)

    # Map regime indices to RGBA colors
    rgba = np.zeros((*attr_mat.shape, 4), dtype=float)
    for ridx in range(8):
        mask_r = attr_mat == ridx
        c = plt.matplotlib.colors.to_rgba(BY_INDEX[ridx]["color"])
        rgba[mask_r] = c
    rgba[attr_mat == -1] = (0.9, 0.9, 0.9, 1.0)   # missing data → light grey

    ax.imshow(
        rgba,
        aspect="auto",
        origin="upper",
        extent=[-0.5, HORIZON_DAYS + 0.5, N_ANALOGS + 0.5, -0.5],
        interpolation="nearest",
    )
    ax.axhline(0.5, color="black", linewidth=1.2, linestyle="--")   # query / analogs separator
    ax.set_yticks([0, N_ANALOGS // 2, N_ANALOGS])
    ax.set_yticklabels(["query", str(N_ANALOGS // 2), str(N_ANALOGS)], fontsize=8)
    ax.set_ylabel(regime, fontsize=10, rotation=0, labelpad=30, va="center")
    ax.set_xlim(-0.5, HORIZON_DAYS + 0.5)
    ax.tick_params(axis="x", labelsize=8)

axes[-1].set_xlabel("Days forward", fontsize=11)

# Shared color legend
handles = [
    plt.matplotlib.patches.Patch(color=BY_INDEX[i]["color"], label=BY_INDEX[i]["name"])
    for i in range(8)
]
fig.legend(
    handles=handles, loc="lower center", ncol=8,
    fontsize=9, bbox_to_anchor=(0.5, -0.01),
)
fig.suptitle(
    "Regime attribution: query (top row) and analogs over 30 days forward",
    fontsize=13,
)
fig.savefig(paths.images_path / "07_analog_regime_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/07_analog_regime_heatmap.png
# :name: fig-07-analog-regime-heatmap
# Lifecycle-attributed regime for the query date (top row, separated by dashed
# line) and each of the 50 analogs (rows below), tracked 30 days forward.
# Each cell is coloured by regime; grey cells indicate missing data.
# ```

# %% [markdown]
# ## Figure 6 — Regime fraction stacked bar chart
#
# For each future day, what fraction of the 50 analog trajectories is attributed
# to each regime?  The query trajectory is excluded from this count.

# %%
fig, axes = plt.subplots(7, 1, figsize=(14, 14), gridspec_kw={"hspace": 0.45})

for ax, regime in zip(axes, WR_NAMES):
    res = analog_results[regime]

    # Attribution matrix for analogs only (exclude query row 0)
    rows = []
    for a_ts in res["analog_ts"]:
        rows.append(regime_trajectory(a_ts))
    attr_mat = np.array(rows)   # shape (N_ANALOGS, 31)

    # Compute fraction per regime per day
    n_valid = np.sum(attr_mat >= 0, axis=0).clip(min=1)
    fracs = np.array([
        np.sum(attr_mat == ridx, axis=0) / n_valid
        for ridx in range(8)
    ])   # shape (8, 31)

    bottom = np.zeros(len(day_offsets))
    for ridx in range(8):
        ax.bar(
            day_offsets, fracs[ridx],
            bottom=bottom,
            color=BY_INDEX[ridx]["color"],
            width=0.85,
            label=BY_INDEX[ridx]["name"],
        )
        bottom += fracs[ridx]

    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, HORIZON_DAYS + 0.5)
    ax.set_ylabel(regime, fontsize=10, rotation=0, labelpad=30, va="center")
    ax.tick_params(labelsize=8)

axes[-1].set_xlabel("Days forward", fontsize=11)

handles = [
    plt.matplotlib.patches.Patch(color=BY_INDEX[i]["color"], label=BY_INDEX[i]["name"])
    for i in range(8)
]
fig.legend(
    handles=handles, loc="lower center", ncol=8,
    fontsize=9, bbox_to_anchor=(0.5, -0.01),
)
fig.suptitle(
    "Regime attribution fractions across 50 analogs, tracked 30 days forward",
    fontsize=13,
)
fig.savefig(paths.images_path / "07_analog_regime_fracs.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/07_analog_regime_fracs.png
# :name: fig-07-analog-regime-fracs
# Stacked bar chart: fraction of the 50 analogs attributed to each regime at
# each future day.  At day 0 the analogs start in a state similar to the
# query date; the chart shows how the ensemble disperses across regimes over
# the following 30 days.
# ```

# %% [markdown]
# ## Figure 7 — Regime attribution heatmap: past 30 days
#
# Same heatmap as Figure 5 but looking *backward*: what regime was each analog
# (and the query) in during the 30 days that led up to day 0?  The columns are
# ordered chronologically (oldest on the left, day 0 on the right).

# %%
fig, axes = plt.subplots(
    7, 1, figsize=(14, 14),
    gridspec_kw={"hspace": 0.35},
)

for ax, regime in zip(axes, WR_NAMES):
    res = analog_results[regime]

    rows = [regime_trajectory_back(res["query_ts"])]
    for a_ts in res["analog_ts"]:
        rows.append(regime_trajectory_back(a_ts))
    # attr_mat_back: col k = k days before reference; flip so oldest is left
    attr_mat_back = np.fliplr(np.array(rows))   # shape (51, 31)

    rgba = np.zeros((*attr_mat_back.shape, 4), dtype=float)
    for ridx in range(8):
        mask_r = attr_mat_back == ridx
        c = plt.matplotlib.colors.to_rgba(BY_INDEX[ridx]["color"])
        rgba[mask_r] = c
    rgba[attr_mat_back == -1] = (0.9, 0.9, 0.9, 1.0)

    ax.imshow(
        rgba,
        aspect="auto",
        origin="upper",
        extent=[-HORIZON_DAYS - 0.5, 0.5, N_ANALOGS + 0.5, -0.5],
        interpolation="nearest",
    )
    ax.axhline(0.5, color="black", linewidth=1.2, linestyle="--")
    ax.set_yticks([0, N_ANALOGS // 2, N_ANALOGS])
    ax.set_yticklabels(["query", str(N_ANALOGS // 2), str(N_ANALOGS)], fontsize=8)
    ax.set_ylabel(regime, fontsize=10, rotation=0, labelpad=30, va="center")
    ax.set_xlim(-HORIZON_DAYS - 0.5, 0.5)
    ax.tick_params(axis="x", labelsize=8)

axes[-1].set_xlabel("Days before", fontsize=11)

handles = [
    plt.matplotlib.patches.Patch(color=BY_INDEX[i]["color"], label=BY_INDEX[i]["name"])
    for i in range(8)
]
fig.legend(
    handles=handles, loc="lower center", ncol=8,
    fontsize=9, bbox_to_anchor=(0.5, -0.01),
)
fig.suptitle(
    "Regime attribution: query (top row) and analogs over the 30 days before day 0",
    fontsize=13,
)
fig.savefig(paths.images_path / "07_analog_regime_heatmap_back.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/07_analog_regime_heatmap_back.png
# :name: fig-07-analog-regime-heatmap-back
# Lifecycle-attributed regime for the query date (top row) and each of the 50
# analogs in the 30 days *before* day 0.  Columns are chronological (oldest on
# the left, day 0 on the right).  Grey cells indicate missing data.
# ```

# %% [markdown]
# ## Figure 8 — Regime fraction stacked bar chart: past 30 days

# %%
fig, axes = plt.subplots(7, 1, figsize=(14, 14), gridspec_kw={"hspace": 0.45})

for ax, regime in zip(axes, WR_NAMES):
    res = analog_results[regime]

    rows = [regime_trajectory_back(a_ts) for a_ts in res["analog_ts"]]
    attr_mat_back = np.array(rows)   # shape (N_ANALOGS, 31); col k = k days before

    # Plot at x = -k so that day 0 is on the right, past on the left
    x_pos   = -day_offsets   # [0, -1, -2, …, -30]
    n_valid = np.sum(attr_mat_back >= 0, axis=0).clip(min=1)
    fracs   = np.array([
        np.sum(attr_mat_back == ridx, axis=0) / n_valid
        for ridx in range(8)
    ])   # shape (8, 31)

    bottom = np.zeros(len(day_offsets))
    for ridx in range(8):
        ax.bar(
            x_pos, fracs[ridx],
            bottom=bottom,
            color=BY_INDEX[ridx]["color"],
            width=0.85,
            label=BY_INDEX[ridx]["name"],
        )
        bottom += fracs[ridx]

    ax.set_ylim(0, 1)
    ax.set_xlim(-HORIZON_DAYS - 0.5, 0.5)
    ax.set_ylabel(regime, fontsize=10, rotation=0, labelpad=30, va="center")
    ax.tick_params(labelsize=8)

axes[-1].set_xlabel("Days before", fontsize=11)

handles = [
    plt.matplotlib.patches.Patch(color=BY_INDEX[i]["color"], label=BY_INDEX[i]["name"])
    for i in range(8)
]
fig.legend(
    handles=handles, loc="lower center", ncol=8,
    fontsize=9, bbox_to_anchor=(0.5, -0.01),
)
fig.suptitle(
    "Regime attribution fractions across 50 analogs, 30 days before day 0",
    fontsize=13,
)
fig.savefig(paths.images_path / "07_analog_regime_fracs_back.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/07_analog_regime_fracs_back.png
# :name: fig-07-analog-regime-fracs-back
# Stacked bar chart: fraction of the 50 analogs attributed to each regime at
# each day in the 30 days before day 0.  Day 0 (right edge) is the selected
# analog state; moving left reveals the regime history that preceded it.
# ```
