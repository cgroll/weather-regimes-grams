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
# # Computing WR Projections from Z500 Fields
#
# This notebook demonstrates how the weather regime index (IWR) is computed
# from a 10-day low-pass filtered Z500 anomaly field, following the method of
# Michel & Rivière (2011).
#
# ## What this script covers
#
# 1. **Projection computation** — reproduced from the author's `WR_read_example.ipynb`
#    (Part 4), using the one bundled example field `Z0500_20250601_00.nc`.
# 2. **Verification** — comparison with the pre-computed value in
#    `wri_projections.csv`.
# 3. **What is needed for a full ERA5 pipeline** — a description of the
#    upstream steps (ERA5 download, climatology, Lanczos filter) required to
#    produce Z500 anomaly fields for arbitrary dates.
#
# ## What this script does NOT cover
#
# **Lifecycle attribution** is not demonstrated here.  The example notebook
# reads pre-computed lifecycle files but does not contain the attribution
# algorithm.  The algorithm (onset / saturation / decay detection from the IWR
# time series) was originally implemented in NCL and is not yet reproduced in
# Python.

# %%
import calendar
import datetime

import netCDF4 as nc
import numpy as np
import pandas as pd

from wr.paths import ProjPaths
from wr.regimes import WR_NAMES, BY_NAME

paths = ProjPaths()

# %% [markdown]
# ## Step 1 — Load normalization weights
#
# `EOFs_WRs.nc` stores a seasonally varying normalization weight `normwgt` at
# 6-hourly resolution for one full leap year (1464 steps).  Before projecting,
# the Z500 anomaly field is divided by the weight corresponding to the
# calendar date of interest.

# %%
with nc.Dataset(paths.grams_eofs) as ds:
    normwgt  = ds.variables["normwgt"][:]   # (1464,) — 6-hourly, one leap year
    nrmtimes = ds.variables["time"][:]      # hours since 1980-01-01 00:00, step=6

print(f"normwgt shape : {normwgt.shape}  (6-hourly steps in a leap year)")
print(f"time range    : {nrmtimes[0]} – {nrmtimes[-1]} h  "
      f"({len(nrmtimes)} steps × 6 h = {len(nrmtimes)*6/24:.0f} days)")

# %% [markdown]
# ## Step 2 — Load regime patterns and standardisation parameters
#
# `Normed_Z0500-patterns_EOFdomain.nc` holds the 7 normalised regime patterns
# on the EOF domain (lat 30–90 °N, lon 80 °W–40 °E, 0.5° resolution →
# 121 × 241 gridpoints).
#
# `WRI_std_params.txt` provides the reference mean and standard deviation of
# the raw projections (computed over 1979–2019) used to standardise them into
# IWR values.

# %%
with nc.Dataset(paths.grams_z500_patterns) as ds:
    patterns = ds.variables["Z0500_mean"][:]   # (7, 121, 241)
    pat_names = ds.ClassNames.split()          # ['AT','ZO','ScTr','AR','EuBL','ScBL','GL']
    lat_eof   = ds.variables["latitude"][:]    # 30.0 … 90.0
    lon_eof   = ds.variables["longitude"][:]   # -80.0 … 40.0

print(f"Patterns shape : {patterns.shape}  — {pat_names}")

with open(paths.grams_std_params) as f:
    f.readline()                                          # description
    std_names = f.readline().split()                      # regime names
    std_mean  = np.array(f.readline().split()[1:], dtype=float)
    std_std   = np.array(f.readline().split()[1:], dtype=float)

std_df = pd.DataFrame({"mean": std_mean, "std": std_std}, index=std_names)
print(f"\nStandardisation parameters (1979-2019 reference):")
print(std_df.T.to_string())

# %% [markdown]
# ## Step 3 — Load the example Z500 field
#
# The bundled file `Z0500_20250601_00.nc` contains a single 10-day
# Lanczos low-pass filtered Z500 anomaly field for 2025-06-01 00 UTC
# (Lanczos filter: 161 points at Δt = 3 h → 240 h cutoff).
# The global field is cropped to the EOF domain before projection.

# %%
target_date = "20250601_00"
dtime = datetime.datetime.strptime(target_date, "%Y%m%d_%H")

with nc.Dataset(paths.grams_example_z500) as ds:
    lon_global = np.arange(ds.domxmin, ds.domxmax + 0.5, 0.5)
    lat_global = np.arange(ds.domymin, ds.domymax + 0.5, 0.5)

    # Crop to EOF domain: lon -80..40, lat 30..90
    i1 = lon_global.tolist().index(-80.0);  i2 = lon_global.tolist().index(40.0) + 1
    j1 = lat_global.tolist().index(30.0);   j2 = lat_global.tolist().index(90.0) + 1

    z500 = ds.variables["Z0"][0, 0, j1:j2, i1:i2].copy()   # (121, 241)

print(f"Z500 field shape (EOF domain): {z500.shape}")
print(f"Value range: {z500.min():.1f} … {z500.max():.1f}  (geopotential metres, anomaly)")

# %% [markdown]
# ## Step 4 — Normalise by seasonal weight
#
# The normwgt array covers 6-hourly steps of a single leap year.
# For a non-leap year, index 236 (= 29 Feb 00 UTC) must be skipped.

# %%
tdiff   = int((dtime - datetime.datetime(dtime.year, 1, 1)).total_seconds() / 3600)
tdiff   = tdiff - tdiff % 6   # snap to 6-h grid

if tdiff >= nrmtimes[236] and not calendar.isleap(dtime.year):
    nj = nrmtimes.tolist().index(tdiff + 24)   # skip 29 Feb slot
else:
    nj = nrmtimes.tolist().index(tdiff)

print(f"Normwgt index: {nj}  →  weight = {normwgt[nj]:.6f}")
z500_norm = z500 / normwgt[nj]

# %% [markdown]
# ## Step 5 — Compute cosine-latitude-weighted projection
#
# The IWR is the cosine-latitude-weighted dot product of the normalised Z500
# anomaly with each regime pattern, then standardised.

# %%
d2r   = np.pi / 180.0
cos2d = np.tile(np.cos(lat_eof * d2r)[:, np.newaxis], (1, len(lon_eof)))
cos2d /= cos2d.sum()   # normalise so weights sum to 1

# Project onto each pattern (patterns are stored in pat_names order)
proj = np.zeros(7)
for w, name in enumerate(pat_names):
    i_wr = WR_NAMES.index(name)
    proj[i_wr] = np.sum(cos2d * patterns[w] * z500_norm)

# Standardise → IWR
iwr = np.zeros(7)
for w, name in enumerate(WR_NAMES):
    iwr[w] = (proj[w] - std_df.loc[name, "mean"]) / std_df.loc[name, "std"]

result = pd.Series(iwr, index=WR_NAMES, name="computed IWR")
print(result.round(4).to_string())

# %% [markdown]
# ## Step 6 — Compare with pre-computed values

# %%
wri = pd.read_csv(paths.wri_csv, parse_dates=["datetime"], index_col="datetime")
ref = wri.loc[pd.Timestamp("2025-06-01 00:00"), WR_NAMES]

comparison = pd.DataFrame({
    "computed": result,
    "reference": ref,
    "diff": result - ref,
})
print(comparison.round(5).to_string())
print(f"\nMax absolute difference: {comparison['diff'].abs().max():.5f}")
print("(Small residuals are expected: Python vs original NCL implementation)")

# %% [markdown]
# ## Regime attribution from IWR projections
#
# The `WR_LCattribution.txt` file has three attribution columns (per
# `fct_wrera_db.py`):
#
# | Column | Name | Description |
# |--------|------|-------------|
# | 2 | `eof_attribution` | not used by the reader function |
# | 3 | `max_wr_index` | regime with the highest IWR — **directly computable** |
# | 4 | `lifecycle_wr_index` | active life-cycle regime — **fully reproducible** |
#
# ### `max_wr_index` — argmax of IWR

# %%
wri     = pd.read_csv(paths.wri_csv,           parse_dates=["datetime"], index_col="datetime")
lc_attr = pd.read_csv(paths.lc_attribution_csv, parse_dates=["datetime"], index_col="datetime")
lc_info = pd.read_csv(paths.lc_info_csv,        parse_dates=["onset", "decay"])

computed_max = wri[WR_NAMES].values.argmax(axis=1) + 1   # 0-based → 1-7 scheme
n_total = len(computed_max)
n_match = (computed_max == lc_attr["max_wr_index"].values).sum()
print(f"max_wr_index exact match: {n_match:,} / {n_total:,}  ({100*n_match/n_total:.4f} %)")

# %% [markdown]
# ### `lifecycle_wr_index` — fully reproducible from lifecycle files
#
# The rule (discovered by checking against the pre-computed values):
#
# 1. A timestep is *active* for regime R if it falls in **[onset, decay]**
#    (both endpoints inclusive) of any life cycle of R.
# 2. When life cycles from different regimes overlap at the same timestep,
#    the one with the **higher IWR** wins.
# 3. Timesteps with no active life cycle get index **0** (no regime).

# %%
import numpy as np

derived_lc  = pd.Series(0, index=wri.index, dtype=int)
winning_iwr = pd.Series(-np.inf, index=wri.index)

for regime in WR_NAMES:
    idx = BY_NAME[regime]["index"]
    for _, lc in lc_info[lc_info["regime"] == regime].iterrows():
        mask     = (wri.index >= lc["onset"]) & (wri.index <= lc["decay"])
        overwrite = mask & (wri[regime] > winning_iwr)
        derived_lc[overwrite]  = idx
        winning_iwr[overwrite] = wri.loc[overwrite, regime]

ref_lc = lc_attr["lifecycle_wr_index"]
n_match_lc = (derived_lc == ref_lc).sum()
print(f"lifecycle_wr_index exact match: {n_match_lc:,} / {n_total:,}  ({100*n_match_lc/n_total:.6f} %)")

# %% [markdown]
# ## What is needed for a full ERA5-based pipeline
#
# The example above uses a single pre-processed Z500 field.  To compute IWR
# for an arbitrary time period from scratch, three upstream steps are required:
#
# ### 1. Download ERA5 Z@500
#
# ERA5 Z@500 at 3-hourly resolution, global 0.5° grid, via the
# Copernicus CDS API (`cdsapi`).  The CDS dataset is
# `reanalysis-era5-pressure-levels`.  Variables: `geopotential` at 500 hPa.
# For the full 1950–present coverage ~100s of GB of data are involved.
# For a demo period (e.g. one season) a few GB suffice.
#
# ### 2. Compute Z500 anomaly from climatology
#
# For each 3-hourly timestep, subtract the 1979–2019 daily climatology
# (available as `CLIM_Z@500_year_1979-2019.nc`).
# `Z0 = Z@500 / 9.81 − climatology`
#
# ### 3. Apply 10-day Lanczos low-pass filter
#
# Apply a 161-point Lanczos LP filter with 240 h cutoff at Δt = 3 h.
# `scipy.signal.firwin` + `scipy.ndimage.convolve1d` along the time axis.
#
# ### 4. Run the projection computation for every timestep
#
# Steps 3–5 above, vectorised over the full time axis → reproduces
# `WRI_projections.txt` and `max_wr_index` exactly.
#
# ### 5. Lifecycle attribution
#
# The lifecycle attribution (onset / saturation / decay detection) was
# originally implemented in NCL.  It is not yet reimplemented in Python.
# For now, the pre-computed `WR_LCattribution.txt` remains authoritative.
