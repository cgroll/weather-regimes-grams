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
# # CF Prediction from Z500 PCs — Model Results
#
# Evaluates Ridge and XGBoost models that predict Germany wind/solar capacity-
# factor anomalies from the leading 20 Z500 EOF principal components.
#
# **Train / test split**
# - Train: 1979–2009 (31 years)
# - Test:  2010–2021 (12 years, held-out future)
#
# **Combinations evaluated**
# - Variables: wind onshore DE, solar PV DE
# - CF averaging window: 1, 2, 5 days
# - Z500 lead time: 0 d (simultaneous), 5 d, 15 d
# - Models: Ridge (standardised PCs, cross-validated α), XGBoost (early stopping)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns

from wr.paths import ProjPaths

paths = ProjPaths()

# %%
scores = pd.read_parquet(paths.cf_model_scores)
preds  = pd.read_parquet(paths.cf_model_predictions)
coefs  = pd.read_parquet(paths.cf_model_coefs)

# Ordered axis labels for heatmaps
LEAD_ORDER   = [0, 5, 15]
WINDOW_ORDER = [1, 2, 5]
VAR_LABELS   = {"wind": "Wind onshore DE", "solar": "Solar PV DE"}
MODEL_LABELS = {"ridge": "Ridge", "xgboost": "XGBoost"}

test_scores = scores[scores.split == "test"].copy()

print("Test-set R² overview:")
print(
    test_scores.pivot_table(
        index=["variable", "window"], columns=["model", "lead"],
        values="r2", aggfunc="first",
    ).round(3).to_string()
)

# %% [markdown]
# ## R² heatmaps — Ridge vs XGBoost

# %%
def r2_heatmap(ax, data: pd.DataFrame, title: str,
               vmin: float = 0.0, vmax: float = None,
               cmap: str = "YlOrRd", annot: bool = True) -> None:
    """Draw a 3×3 heatmap of test R² on ax."""
    pivot = (
        data.pivot(index="window", columns="lead", values="r2")
        .reindex(index=WINDOW_ORDER, columns=LEAD_ORDER)
    )
    if vmax is None:
        vmax = min(pivot.values.max() * 1.05, 1.0)
    sns.heatmap(
        pivot, ax=ax,
        vmin=vmin, vmax=vmax, cmap=cmap,
        annot=annot, fmt=".3f", annot_kws={"size": 10},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Test R²", "shrink": 0.85},
    )
    ax.set_xlabel("Lead time (days)", fontsize=10)
    ax.set_ylabel("CF avg window (days)", fontsize=10)
    ax.set_title(title, fontsize=11, pad=6)
    ax.tick_params(labelsize=9)


fig, axes = plt.subplots(
    2, 3, figsize=(16, 9),
    gridspec_kw={"hspace": 0.45, "wspace": 0.35},
)

# Shared colour scale across all 6 panels
global_vmax = round(test_scores.r2.clip(lower=0).max() * 1.05, 2)

for row, var in enumerate(["wind", "solar"]):
    for col, model in enumerate(["ridge", "xgboost"]):
        ax = axes[row, col]
        subset = test_scores[(test_scores.variable == var) &
                             (test_scores.model == model)]
        r2_heatmap(
            ax, subset,
            title=f"{VAR_LABELS[var]} — {MODEL_LABELS[model]}",
            vmax=global_vmax,
        )

    # Difference panel: XGBoost − Ridge
    ax_diff = axes[row, 2]
    r_ridge = test_scores[(test_scores.variable == var) & (test_scores.model == "ridge")]
    r_xgb   = test_scores[(test_scores.variable == var) & (test_scores.model == "xgboost")]
    diff_df = r_xgb[["lead", "window", "r2"]].copy()
    diff_df["r2"] = (
        r_xgb["r2"].values - r_ridge["r2"].values
    )
    dmax = max(abs(diff_df.r2).max(), 0.005)
    diff_pivot = (
        diff_df.pivot(index="window", columns="lead", values="r2")
        .reindex(index=WINDOW_ORDER, columns=LEAD_ORDER)
    )
    sns.heatmap(
        diff_pivot, ax=ax_diff,
        vmin=-dmax, vmax=dmax, cmap="RdBu_r",
        annot=True, fmt="+.3f", annot_kws={"size": 10},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "ΔR² (XGB − Ridge)", "shrink": 0.85},
    )
    ax_diff.set_xlabel("Lead time (days)", fontsize=10)
    ax_diff.set_ylabel("CF avg window (days)", fontsize=10)
    ax_diff.set_title(f"{VAR_LABELS[var]} — XGBoost − Ridge", fontsize=11, pad=6)
    ax_diff.tick_params(labelsize=9)

fig.suptitle(
    "Test-set R² for CF anomaly prediction from Z500 PCs  (2010–2021)\n"
    "Columns: Ridge | XGBoost | Difference",
    fontsize=13,
)

IMG_R2 = paths.images_path / "23_cf_model_r2.png"
fig.savefig(IMG_R2, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {IMG_R2}")

# %% [markdown]
# ```{figure} ../../output/images/23_cf_model_r2.png
# :name: fig-23-cf-model-r2
# Test-set R² for predicting Germany wind onshore (top) and solar PV (bottom)
# CF anomalies from 20 Z500 EOF principal components.  Each cell shows the R²
# for a specific CF averaging window (rows: 1, 2, 5 days) and Z500 lead time
# (columns: 0, 5, 15 days).  Left: Ridge; centre: XGBoost; right: XGBoost minus
# Ridge (positive = XGBoost gains over Ridge).
# ```

# %% [markdown]
# ## Feature importance

# %%
PC_ORDER = [f"PC{k:02d}" for k in range(1, 21)]

fig, axes = plt.subplots(
    2, 2, figsize=(16, 10),
    gridspec_kw={"hspace": 0.45, "wspace": 0.35},
)

for row, var in enumerate(["wind", "solar"]):
    for col, model in enumerate(["ridge", "xgboost"]):
        ax = axes[row, col]
        sub = coefs[(coefs.variable == var) & (coefs.model == model)]

        # Average importance across all (lead, window) combinations
        mean_imp = (
            sub.groupby("feature")["value"]
            .agg(mean="mean", std="std")
            .reindex(PC_ORDER)
        )

        x = np.arange(len(PC_ORDER))
        if model == "ridge":
            # Show mean absolute coefficient; error bars = std across combos
            bars = ax.bar(x, mean_imp["mean"].abs(),
                          yerr=mean_imp["std"],
                          color="steelblue", alpha=0.8,
                          error_kw={"elinewidth": 0.8, "capsize": 2})
            ax.set_ylabel("|Ridge coef| (std-units → CF anomaly)", fontsize=9)
        else:
            bars = ax.bar(x, mean_imp["mean"],
                          yerr=mean_imp["std"],
                          color="tomato", alpha=0.8,
                          error_kw={"elinewidth": 0.8, "capsize": 2})
            ax.set_ylabel("XGBoost gain importance (mean ± std)", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(PC_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("EOF principal component", fontsize=9)
        ax.set_title(
            f"{VAR_LABELS[var]} — {MODEL_LABELS[model]}\n"
            f"(mean ± std over {len(LEAD_ORDER) * len(WINDOW_ORDER)} combos)",
            fontsize=10,
        )
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=9)

fig.suptitle(
    "Feature importance: which Z500 PCs matter most for CF prediction?",
    fontsize=13,
)

IMG_IMP = paths.images_path / "23_cf_model_importance.png"
fig.savefig(IMG_IMP, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {IMG_IMP}")

# %% [markdown]
# ```{figure} ../../output/images/23_cf_model_importance.png
# :name: fig-23-cf-model-importance
# Mean feature importance across all 9 lead × window combinations.  Left:
# absolute Ridge coefficient (standardised PCs → CF anomaly). Right: XGBoost
# gain importance.  Error bars show the standard deviation across the 9
# combinations.
# ```

# %% [markdown]
# ## Predicted vs actual — best combination

# %%
# Identify the (lead, window) combination with highest test R² for each variable
best = (
    test_scores[test_scores.model == "ridge"]
    .sort_values("r2", ascending=False)
    .groupby("variable")
    .first()
    .reset_index()[["variable", "lead", "window", "r2"]]
)
print("Best Ridge combination per variable:")
print(best.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                          gridspec_kw={"wspace": 0.35})

for ax, (_, row) in zip(axes, best.iterrows()):
    var, lag, win = row.variable, int(row.lead), int(row.window)

    # Retrieve test predictions for both models
    sel = preds[(preds.variable == var) & (preds.lead == lag) &
                (preds.window == win)].copy()
    sel["date"] = pd.to_datetime(sel["date"])

    for model, color, label in [("ridge", "steelblue", "Ridge"),
                                  ("xgboost", "tomato", "XGBoost")]:
        m = sel[sel.model == model].sort_values("date")
        r2 = test_scores.loc[
            (test_scores.variable == var) & (test_scores.model == model) &
            (test_scores.lead == lag) & (test_scores.window == win),
            "r2",
        ].values[0]
        ax.scatter(m["actual"], m["predicted"], alpha=0.25, s=4,
                   color=color, label=f"{label}  R²={r2:.3f}")

    # 1:1 line
    lo = sel["actual"].min()
    hi = sel["actual"].max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel(f"Actual CF anomaly ({win}d avg)", fontsize=10)
    ax.set_ylabel(f"Predicted CF anomaly", fontsize=10)
    ax.set_title(
        f"{VAR_LABELS[var]}\nlead={lag} d, window={win} d",
        fontsize=11,
    )
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(alpha=0.3)

fig.suptitle("Predicted vs actual CF anomaly — test set (2010–2021)", fontsize=13)

IMG_SCATTER = paths.images_path / "23_cf_model_scatter.png"
fig.savefig(IMG_SCATTER, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {IMG_SCATTER}")

# %% [markdown]
# ```{figure} ../../output/images/23_cf_model_scatter.png
# :name: fig-23-cf-model-scatter
# Predicted vs actual CF anomaly on the test set (2010–2021) for the best-
# performing combination per variable (highest Ridge test R²).  Blue: Ridge;
# red: XGBoost.  Dashed line: perfect prediction.
# ```
