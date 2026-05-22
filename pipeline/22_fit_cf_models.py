"""
Fit Ridge and XGBoost models to predict Germany wind/solar CF anomalies
from Z500 EOF principal components.

Combinations
------------
  variable : wind_onshore | solar_pv
  window   : 1, 2, 5 days  (forward mean of CF anomaly)
  lead     : 0, 5, 15 days (Z500 leads CF by this many days)
  model    : ridge | xgboost

Train / test split (defined on the predictor date, i.e. the Z500 day)
  Train : 1979-01-01 – 2009-12-31  (~31 years)
  Test  : 2010-01-01 – 2021-12-31  (~12 years)

Outputs (all in data/processed/)
---------------------------------
  cf_model_scores.parquet      R², RMSE, Pearson r  per combination
  cf_model_predictions.parquet Test-set predicted vs actual (long format)
  cf_model_coefs.parquet       Ridge standardised coefficients +
                               XGBoost gain importances (long format)
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from xgboost import XGBRegressor

from wr.paths import ProjPaths

warnings.filterwarnings("ignore", category=UserWarning)

paths = ProjPaths()

PECD_PATH  = "/home/chris/research/world-of-energy/data/processed/pecd/pecd_regions.parquet"
TRAIN_END  = pd.Timestamp("2009-12-31")
TEST_START = pd.Timestamp("2010-01-01")
WINDOWS    = [1, 2, 5]
LAGS       = [0, 5, 15]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fourier_climatology(values: np.ndarray, n_harmonics: int = 4) -> np.ndarray:
    N = len(values)
    t = np.arange(N)
    omega = 2 * np.pi * t / N
    smooth = np.full(N, np.mean(values), dtype=float)
    for h in range(1, n_harmonics + 1):
        cc = np.mean(values * np.cos(h * omega)) * 2
        sc = np.mean(values * np.sin(h * omega)) * 2
        smooth += cc * np.cos(h * omega) + sc * np.sin(h * omega)
    return smooth


def compute_cf_anomaly(series: pd.Series) -> pd.Series:
    g = series.groupby([series.index.month, series.index.day])
    raw_clim = g.mean()
    smooth   = fourier_climatology(raw_clim.values, n_harmonics=4)
    clim_map = dict(zip(raw_clim.index, smooth))
    clim_vals = np.array([clim_map[(m, d)]
                          for m, d in zip(series.index.month, series.index.day)])
    return series - clim_vals


def forward_mean(arr: np.ndarray, w: int) -> np.ndarray:
    """result[i] = mean(arr[i:i+w]).  Length = len(arr) - w + 1."""
    if w == 1:
        return arr.copy()
    return np.convolve(arr, np.ones(w) / w, mode="valid")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading Z500 PCs …")
pcs_df = pd.read_parquet(paths.z500_pcs)
pcs_df.index = pd.to_datetime(pcs_df.index).normalize()

print("Loading PECD …")
pecd = pd.read_parquet(PECD_PATH)
wind_h  = pecd[("wind_power_generation_onshore",       "capacity_factor_ratio", "DE")]
solar_h = pecd[("solar_photovoltaic_power_generation", "capacity_factor_ratio", "DE")]

wind_d  = wind_h.resample("D").mean()
solar_d = solar_h.resample("D").mean()

# CF anomalies (Fourier climatology, same as script 20)
print("Computing CF anomalies …")
wind_anom  = compute_cf_anomaly(wind_d["1979":"2021"])
solar_anom = compute_cf_anomaly(solar_d["1979":"2021"])

# Align on common dates
common = pcs_df.index.intersection(wind_anom.index)
print(f"Common dates: {common[0].date()} → {common[-1].date()}, n={len(common)}")

pc_vals    = pcs_df.loc[common].values.astype(np.float32)   # (T, 20)
wind_vals  = wind_anom.reindex(common).values.astype(np.float32)
solar_vals = solar_anom.reindex(common).values.astype(np.float32)
T          = len(common)
PC_NAMES   = pcs_df.columns.tolist()

VARIABLES = {"wind": wind_vals, "solar": solar_vals}

# ---------------------------------------------------------------------------
# Model fitting loop
# ---------------------------------------------------------------------------

rows_scores = []
rows_preds  = []
rows_coefs  = []

for var_name, cf_vals in VARIABLES.items():
    for window in WINDOWS:
        cf_fwd = forward_mean(cf_vals, window)   # length T - window + 1

        for lag in LAGS:
            n_pairs = len(cf_fwd) - lag          # valid predictor-target pairs
            X_all = pc_vals[:n_pairs]
            y_all = cf_fwd[lag:]
            dates = common[:n_pairs]

            train_mask = dates <= TRAIN_END
            test_mask  = dates >= TEST_START

            X_tr, y_tr = X_all[train_mask], y_all[train_mask]
            X_te, y_te = X_all[test_mask],  y_all[test_mask]
            dates_te   = dates[test_mask]

            tag = f"{var_name} w={window}d lag={lag:2d}d"
            print(f"  {tag}  train={train_mask.sum():5d}  test={test_mask.sum():5d}")

            # ── Ridge ────────────────────────────────────────────────────────
            scaler   = StandardScaler().fit(X_tr)
            X_tr_s   = scaler.transform(X_tr)
            X_te_s   = scaler.transform(X_te)

            # RidgeCV with LOO efficient path; alphas span several decades
            ridge = RidgeCV(alphas=np.logspace(-2, 5, 60), fit_intercept=True)
            ridge.fit(X_tr_s, y_tr)

            yp_tr_r = ridge.predict(X_tr_s)
            yp_te_r = ridge.predict(X_te_s)

            for split, y_t, y_p in [("train", y_tr, yp_tr_r),
                                     ("test",  y_te, yp_te_r)]:
                rows_scores.append(dict(
                    model="ridge", variable=var_name, lead=lag, window=window,
                    split=split,
                    r2=r2_score(y_t, y_p),
                    rmse=float(np.sqrt(np.mean((y_t - y_p) ** 2))),
                    pearson_r=float(pearsonr(y_t, y_p).statistic),
                    n=len(y_t),
                    best_alpha=float(ridge.alpha_),
                ))

            for k, coef in enumerate(ridge.coef_):
                rows_coefs.append(dict(
                    model="ridge", variable=var_name, lead=lag, window=window,
                    feature=PC_NAMES[k], value=float(coef),
                ))

            for date, actual, pred in zip(dates_te, y_te, yp_te_r):
                rows_preds.append(dict(
                    date=date, model="ridge", variable=var_name,
                    lead=lag, window=window,
                    actual=float(actual), predicted=float(pred),
                ))

            # ── XGBoost ──────────────────────────────────────────────────────
            # Hold out the last 15 % of training data for early stopping
            val_size    = max(int(len(y_tr) * 0.15), 365)
            n_main      = len(y_tr) - val_size
            X_tr_m, y_tr_m = X_tr[:n_main], y_tr[:n_main]
            X_val,  y_val  = X_tr[n_main:], y_tr[n_main:]

            xgb = XGBRegressor(
                n_estimators=500,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                early_stopping_rounds=30,
                random_state=42,
                verbosity=0,
            )
            xgb.fit(
                X_tr_m, y_tr_m,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            yp_tr_x = xgb.predict(X_tr)
            yp_te_x = xgb.predict(X_te)

            for split, y_t, y_p in [("train", y_tr, yp_tr_x),
                                     ("test",  y_te, yp_te_x)]:
                rows_scores.append(dict(
                    model="xgboost", variable=var_name, lead=lag, window=window,
                    split=split,
                    r2=r2_score(y_t, y_p),
                    rmse=float(np.sqrt(np.mean((y_t - y_p) ** 2))),
                    pearson_r=float(pearsonr(y_t, y_p).statistic),
                    n=len(y_t),
                    best_alpha=float("nan"),
                ))

            for k, imp in enumerate(xgb.feature_importances_):
                rows_coefs.append(dict(
                    model="xgboost", variable=var_name, lead=lag, window=window,
                    feature=PC_NAMES[k], value=float(imp),
                ))

            for date, actual, pred in zip(dates_te, y_te, yp_te_x):
                rows_preds.append(dict(
                    date=date, model="xgboost", variable=var_name,
                    lead=lag, window=window,
                    actual=float(actual), predicted=float(pred),
                ))

            r2_ridge = rows_scores[-4]["r2"] if rows_scores else 0
            r2_xgb   = rows_scores[-2]["r2"] if rows_scores else 0
            print(f"    Ridge R²={rows_scores[-3]['r2']:.3f}  "
                  f"XGBoost R²={rows_scores[-1]['r2']:.3f}  "
                  f"(α={ridge.alpha_:.2e})")

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

paths.processed_data_path.mkdir(parents=True, exist_ok=True)

scores_df = pd.DataFrame(rows_scores)
scores_df.to_parquet(paths.cf_model_scores)
print(f"\nScores → {paths.cf_model_scores}  {scores_df.shape}")

preds_df = pd.DataFrame(rows_preds)
preds_df.to_parquet(paths.cf_model_predictions)
print(f"Predictions → {paths.cf_model_predictions}  {preds_df.shape}")

coefs_df = pd.DataFrame(rows_coefs)
coefs_df.to_parquet(paths.cf_model_coefs)
print(f"Coefficients → {paths.cf_model_coefs}  {coefs_df.shape}")

# Quick summary table
print("\n── Test-set R² summary ──")
summary = (
    scores_df[scores_df.split == "test"]
    .pivot_table(index=["variable", "window"], columns=["model", "lead"],
                 values="r2", aggfunc="first")
    .round(3)
)
print(summary.to_string())
