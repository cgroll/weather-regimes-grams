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
# # Golden Field: Grams vs ERA5 + WeatherBench2 Climatology
#
# The Grams Zenodo package ships one pre-processed field,
# `Z0500_20250601_00.nc` — the 10-day Lanczos LP-filtered Z500 anomaly for
# 2025-06-01 00 UTC (the "golden field").  Script 05 shows that projecting
# this field reproduces the published IWR exactly.
#
# This notebook recomputes the golden field from raw ERA5 Z500 (script 26)
# using the WeatherBench2 seasonal climatology (script 24) and compares the
# two fields grid point by grid point on the EOF domain
# (30–90 °N, 80 °W–40 °E, 0.5°, 121 × 241 points).
#
# Recomputation:
#
# 1. ERA5 Z500, 3-hourly, 2025-05-20 → 2025-06-13.
# 2. Subtract the WB2 day-of-year × hour climatology (1990–2019); 03/09/15/21
#    UTC use the mean of the neighbouring 6-hourly fields.
# 3. 161-point Lanczos LP filter (cutoff 240 h) → value at 2025-06-01 00 UTC.
#
# For context the year-round Grams mean (as used in script 06) is evaluated
# alongside.  All results are also written to a PDF report in
# `output/reports/`.

# %%
import calendar
import textwrap

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

from wr.paths import ProjPaths
from wr.regimes import WR_NAMES

paths = ProjPaths()
paths.ensure_directories()

TARGET = pd.Timestamp("2025-06-01 00:00")
G = 9.80665

# %% [markdown]
# ## Step 1 — Load the golden field (EOF domain)

# %%
with nc.Dataset(paths.grams_example_z500) as ds:
    lon_global = np.arange(ds.domxmin, ds.domxmax + 0.5, 0.5)
    lat_global = np.arange(ds.domymin, ds.domymax + 0.5, 0.5)
    i1 = lon_global.tolist().index(-80.0); i2 = lon_global.tolist().index(40.0) + 1
    j1 = lat_global.tolist().index(30.0);  j2 = lat_global.tolist().index(90.0) + 1
    golden = np.asarray(ds.variables["Z0"][0, 0, j1:j2, i1:i2], dtype=float)   # S→N

lat = lat_global[j1:j2]   # 30 … 90
lon = lon_global[i1:i2]   # -80 … 40
print(f"Golden field: {golden.shape}")

# %% [markdown]
# ## Step 2 — Recompute the field from ERA5

# %%
era5  = xr.open_dataset(paths.era5_z500_nc("2025-06-01", "2025-06-01"))
z0500 = era5["z0500"].sortby("latitude")          # S→N
assert np.allclose(z0500.latitude.values, lat) and np.allclose(z0500.longitude.values, lon)
times = pd.DatetimeIndex(z0500.time.values)
print(f"ERA5: {z0500.shape}  ({times[0]} → {times[-1]})")

wb = xr.open_zarr(paths.wb_z500_climatology_6h)["z500"].sortby("latitude")
assert np.allclose(wb.latitude.values, lat)
wb_hours = list(wb.hour.values)
clim_wb  = wb.values   # (4, 366, 121, 241)


def _wb_field(ts: pd.Timestamp) -> np.ndarray:
    return clim_wb[wb_hours.index(ts.hour), ts.dayofyear - 1]


def wb_clim_at(ts: pd.Timestamp) -> np.ndarray:
    if ts.hour % 6 == 0:
        return _wb_field(ts)
    return 0.5 * (_wb_field(ts - pd.Timedelta("3h")) + _wb_field(ts + pd.Timedelta("3h")))


with nc.Dataset(paths.grams_clim_z500) as ds:
    c_lat = ds.variables["latitude"][:]
    c_lon = ds.variables["longitude"][:]
    clim_year = np.asarray(ds.variables["Z@500"][:] / G)[
        np.ix_((c_lat >= 30.0) & (c_lat <= 90.0), (c_lon >= -80.0) & (c_lon <= 40.0))
    ]   # S→N


def _lanczos_lp_weights(n: int, cutoff_ts: float) -> np.ndarray:
    """Lanczos LP filter weights (n odd, cutoff in timesteps)."""
    m = (n - 1) // 2
    k = np.arange(-m, m + 1, dtype=float)
    fc = 1.0 / cutoff_ts
    k_nz = np.where(k == 0, 1.0, k)
    w     = np.sin(2 * np.pi * fc * k_nz) / (np.pi * k_nz)
    w[m]  = 2 * fc
    sigma     = np.sin(np.pi * k_nz / m) / (np.pi * k_nz / m)
    sigma[m]  = 1.0
    w *= sigma
    w /= w.sum()
    return w


weights = _lanczos_lp_weights(161, 240 / 3)
m       = (len(weights) - 1) // 2
i_tgt   = times.get_loc(TARGET)
win     = slice(i_tgt - m, i_tgt + m + 1)
assert i_tgt - m >= 0 and i_tgt + m < len(times), "ERA5 window too short for the filter"

z_win = z0500.values[win]
clim_seasonal = np.stack([wb_clim_at(t) for t in times[win]])

recomp      = np.einsum("t,tij->ij", weights, z_win - clim_seasonal)          # WB2 seasonal
recomp_year = np.einsum("t,tij->ij", weights, z_win - clim_year[np.newaxis])  # year-round (context)

# %% [markdown]
# ## Step 3 — Value range of the golden field and error metrics
#
# Metrics are given unweighted (every grid point counts equally) and
# cos(latitude)-weighted (area-weighted — the weighting used in the IWR
# projection, so it reflects what matters for the regime index).

# %%
cosw = np.cos(np.deg2rad(lat))[:, np.newaxis] * np.ones_like(golden)
cosw /= cosw.sum()


def field_stats(f: np.ndarray) -> dict:
    return {
        "min": f.min(), "max": f.max(), "range": f.max() - f.min(),
        "mean": f.mean(), "std": f.std(),
        "p05": np.percentile(f, 5), "p95": np.percentile(f, 95),
        "mean |value|": np.abs(f).mean(),
    }


def error_metrics(f: np.ndarray, ref: np.ndarray) -> dict:
    d = f - ref
    return {
        "bias":         d.mean(),
        "MAE":          np.abs(d).mean(),
        "RMSE":         np.sqrt((d ** 2).mean()),
        "max |diff|":   np.abs(d).max(),
        "MAE (cos-w)":  (cosw * np.abs(d)).sum(),
        "RMSE (cos-w)": np.sqrt((cosw * d ** 2).sum()),
        "pattern r":    np.corrcoef(f.ravel(), ref.ravel())[0, 1],
    }


stats_tbl = pd.DataFrame({
    "Grams golden":         field_stats(golden),
    "ERA5 − WB2 seasonal":  field_stats(recomp),
    "ERA5 − year-round":    field_stats(recomp_year),
}).T
err_tbl = pd.DataFrame({
    "ERA5 − WB2 seasonal": error_metrics(recomp, golden),
    "ERA5 − year-round":   error_metrics(recomp_year, golden),
}).T

g_range, g_std = stats_tbl.loc["Grams golden", "range"], stats_tbl.loc["Grams golden", "std"]
rel_tbl = pd.DataFrame({
    "MAE / range": err_tbl["MAE"] / g_range,
    "RMSE / range": err_tbl["RMSE"] / g_range,
    "RMSE / std": err_tbl["RMSE"] / g_std,
})

print("Field statistics (gpm)")
print(stats_tbl.round(1).to_string())
print("\nError vs golden field (gpm)")
print(err_tbl.round(3).to_string())
print("\nRelative to the golden field")
print((rel_tbl * 100).round(2).astype(str).add(" %").to_string())

# %% [markdown]
# ## Step 4 — Maps: golden field, recomputed field, difference

# %%
proj = ccrs.LambertConformal(central_longitude=-20, central_latitude=60,
                             standard_parallels=(45, 75))
extent = [-80, 40, 30, 90]
lon2d, lat2d = np.meshgrid(lon, lat)


def _map_ax(ax, field, title, cmap, norm, label):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    pm = ax.pcolormesh(lon2d, lat2d, field, cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), shading="auto", rasterized=True)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="0.25")
    ax.gridlines(linewidth=0.3, color="0.6", alpha=0.6)
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(pm, ax=ax, orientation="horizontal", pad=0.04, shrink=0.9)
    cb.set_label(label, fontsize=9)


vmax_f = np.ceil(max(np.abs(golden).max(), np.abs(recomp).max()) / 10) * 10
norm_f = TwoSlopeNorm(vcenter=0, vmin=-vmax_f, vmax=vmax_f)
diff   = recomp - golden
vmax_d = np.ceil(np.percentile(np.abs(diff), 99.5))
norm_d = TwoSlopeNorm(vcenter=0, vmin=-vmax_d, vmax=vmax_d)

fig_maps, axes = plt.subplots(1, 3, figsize=(17, 6.2), subplot_kw={"projection": proj})
_map_ax(axes[0], golden, "Grams golden field\n(LP-filtered Z500 anomaly)", "RdBu_r", norm_f, "gpm")
_map_ax(axes[1], recomp, "Recomputed: ERA5 − WB2 seasonal clim\n(LP-filtered)", "RdBu_r", norm_f, "gpm")
_map_ax(axes[2], diff, "Difference: recomputed − golden", "PuOr_r", norm_d, "gpm")
fig_maps.suptitle(f"Z500 LP-filtered anomaly, {TARGET:%Y-%m-%d %H} UTC", fontsize=12)
fig_maps.tight_layout()
fig_maps.savefig(paths.images_path / "27_golden_field_maps.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/27_golden_field_maps.png
# :name: fig-27-golden-field-maps
# Left: the Grams golden field.  Middle: the same quantity recomputed from
# ERA5 with the WeatherBench2 seasonal climatology.  Right: difference per
# grid point (note the different colour scale).
# ```

# %% [markdown]
# ## Step 5 — Scatter: golden vs recomputed values

# %%
fig_sc, ax = plt.subplots(figsize=(6.5, 6.5))
hb = ax.hexbin(golden.ravel(), recomp.ravel(), gridsize=120, bins="log",
               cmap="Blues", mincnt=1)
lims = [-vmax_f, vmax_f]
ax.plot(lims, lims, color="0.2", lw=1, ls="--", label="1 : 1")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_aspect("equal")
ax.set_xlabel("Grams golden field (gpm)")
ax.set_ylabel("Recomputed ERA5 − WB2 seasonal (gpm)")
e = err_tbl.loc["ERA5 − WB2 seasonal"]
ax.text(0.03, 0.97,
        f"r = {e['pattern r']:.4f}\nMAE = {e['MAE']:.2f} gpm\nRMSE = {e['RMSE']:.2f} gpm\n"
        f"bias = {e['bias']:+.2f} gpm\nn = {golden.size:,} grid points",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="0.8"))
ax.legend(loc="lower right", fontsize=9)
plt.colorbar(hb, ax=ax, shrink=0.8, label="grid points per bin (log)")
ax.grid(color="0.9", lw=0.5)
fig_sc.tight_layout()
fig_sc.savefig(paths.images_path / "27_golden_field_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/27_golden_field_scatter.png
# :name: fig-27-golden-field-scatter
# Golden-field values against recomputed values for all 29 161 grid points of
# the EOF domain.  The dashed line marks perfect agreement.
# ```

# %% [markdown]
# ## Step 6 — Resulting IWR
#
# Both fields are projected exactly as in script 05 (normwgt → cos-lat
# weighted projection → standardisation).  The published Grams value for
# 2025-06-01 00 UTC is shown as reference.

# %%
with nc.Dataset(paths.grams_eofs) as ds:
    normwgt  = ds.variables["normwgt"][:]
    nrmtimes = ds.variables["time"][:]
tdiff = int((TARGET - pd.Timestamp(f"{TARGET.year}-01-01")).total_seconds() / 3600)
tdiff -= tdiff % 6
if tdiff >= nrmtimes[236] and not calendar.isleap(TARGET.year):
    nj = int(np.where(nrmtimes == tdiff + 24)[0][0])
else:
    nj = int(np.where(nrmtimes == tdiff)[0][0])

with nc.Dataset(paths.grams_z500_patterns) as ds:
    patterns  = ds.variables["Z0500_mean"][:]
    pat_names = ds.ClassNames.split()

with open(paths.grams_std_params) as f:
    f.readline()
    std_names = f.readline().split()
    std_mean  = np.array(f.readline().split()[1:], dtype=float)
    std_std   = np.array(f.readline().split()[1:], dtype=float)

cos2d = np.cos(np.deg2rad(lat))[:, np.newaxis] * np.ones_like(golden)
cos2d /= cos2d.sum()


def iwr_of(field: np.ndarray) -> pd.Series:
    out = {}
    for w, name in enumerate(pat_names):
        i_std = std_names.index(name)
        p = (cos2d * patterns[w] * field / normwgt[nj]).sum()
        out[name] = (p - std_mean[i_std]) / std_std[i_std]
    return pd.Series(out)[WR_NAMES]


ref = pd.read_csv(paths.wri_csv, parse_dates=["datetime"], index_col="datetime")
iwr_tbl = pd.DataFrame({
    "Grams published":       ref.loc[TARGET, WR_NAMES].astype(float),
    "golden field":          iwr_of(golden),
    "ERA5 − WB2 seasonal":   iwr_of(recomp),
    "ERA5 − year-round":     iwr_of(recomp_year),
})
iwr_tbl["Δ WB2 seasonal"] = iwr_tbl["ERA5 − WB2 seasonal"] - iwr_tbl["golden field"]
iwr_tbl["Δ year-round"]   = iwr_tbl["ERA5 − year-round"] - iwr_tbl["golden field"]
print(iwr_tbl.round(3).to_string())
print(f"\nMAE over 7 regimes — WB2 seasonal: {iwr_tbl['Δ WB2 seasonal'].abs().mean():.3f}   "
      f"year-round: {iwr_tbl['Δ year-round'].abs().mean():.3f}")
print(f"Max-IWR regime — golden: {iwr_tbl['golden field'].idxmax()}   "
      f"WB2 seasonal: {iwr_tbl['ERA5 − WB2 seasonal'].idxmax()}")

# %%
fig_iwr, ax = plt.subplots(figsize=(10, 4.8))
x  = np.arange(len(WR_NAMES))
bw = 0.26
series = [
    ("golden field",        "0.35",      "Grams golden field"),
    ("ERA5 − WB2 seasonal", "steelblue", "ERA5 − WB2 seasonal"),
    ("ERA5 − year-round",   "tab:red",   "ERA5 − year-round (context)"),
]
for i, (col, color, label) in enumerate(series):
    ax.bar(x + (i - 1) * bw, iwr_tbl[col], bw * 0.92, color=color, label=label)
ax.axhline(0, color="k", lw=0.6)
ax.axhline(1, color="k", lw=0.6, ls="--")
ax.set_xticks(x)
ax.set_xticklabels(WR_NAMES)
ax.set_ylabel("IWR")
ax.set_title(f"IWR on {TARGET:%Y-%m-%d %H} UTC from golden vs recomputed field", fontsize=11)
for xi, d in zip(x, iwr_tbl["Δ WB2 seasonal"]):
    ax.annotate(f"Δ {d:+.3f}", (xi, ax.get_ylim()[0]), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=8, color="0.25")
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", color="0.9", lw=0.5)
fig_iwr.tight_layout()
fig_iwr.savefig(paths.images_path / "27_golden_field_iwr.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/27_golden_field_iwr.png
# :name: fig-27-golden-field-iwr
# IWR from the golden field (grey), from the recomputed field with the WB2
# seasonal climatology (blue) and, for context, with the year-round Grams
# mean (red).  Δ labels: WB2-seasonal minus golden.
# ```

# %% [markdown]
# ## Step 7 — Write the PDF report

# %%
def _table_page(pdf, title: str, blocks: list[tuple[str, pd.DataFrame, str]], notes: str = ""):
    fig = plt.figure(figsize=(11.69, 8.27))   # A4 landscape
    fig.text(0.05, 0.94, title, fontsize=16, weight="bold")
    y = 0.88
    for subtitle, df, fmt in blocks:
        fig.text(0.05, y, subtitle, fontsize=11, weight="bold")
        # Row labels go into the first column so they stay inside the axes
        cells = [[str(idx)] + [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                               for v in row]
                 for idx, row in zip(df.index, df.values)]
        h = 0.035 * (len(df) + 1)
        ax = fig.add_axes([0.05, y - 0.01 - h, 0.8, h])
        ax.axis("off")
        tbl = ax.table(cellText=cells, colLabels=[""] + list(df.columns),
                       loc="upper left", cellLoc="right")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        for r in range(len(df) + 1):
            tbl[r, 0].set_text_props(ha="left")
        tbl.auto_set_column_width(0)
        y -= h + 0.08
    if notes:
        wrapped = "\n".join(textwrap.fill(p, 150) for p in notes.split("\n"))
        fig.text(0.05, max(y, 0.04), wrapped, fontsize=9, va="top", linespacing=1.5)
    pdf.savefig(fig)
    plt.close(fig)


e  = err_tbl.loc["ERA5 − WB2 seasonal"]
gs = stats_tbl.loc["Grams golden"]
summary = (
    f"Golden field (Z0500_20250601_00.nc, EOF domain 30–90°N, 80°W–40°E, {golden.size:,} points): "
    f"values from {gs['min']:.1f} to {gs['max']:.1f} gpm (range {gs['range']:.1f} gpm, std {gs['std']:.1f} gpm).\n"
    f"Recomputed from ERA5 with the WeatherBench2 seasonal climatology (1990–2019): "
    f"MAE {e['MAE']:.2f} gpm ({100 * e['MAE'] / gs['range']:.2f} % of range), "
    f"RMSE {e['RMSE']:.2f} gpm ({100 * e['RMSE'] / gs['range']:.2f} % of range, "
    f"{100 * e['RMSE'] / gs['std']:.1f} % of std), bias {e['bias']:+.2f} gpm, pattern r = {e['pattern r']:.4f}.\n"
    f"IWR: mean |Δ| over the 7 regimes = {iwr_tbl['Δ WB2 seasonal'].abs().mean():.3f} "
    f"(year-round mean: {iwr_tbl['Δ year-round'].abs().mean():.3f})."
)

pdf_path = paths.reports_path / "27_golden_field_report.pdf"
with PdfPages(pdf_path) as pdf:
    _table_page(
        pdf,
        f"Golden field vs recomputed field — {TARGET:%Y-%m-%d %H} UTC",
        [
            ("Field statistics (gpm)", stats_tbl, "{:.1f}"),
            ("Errors vs golden field (gpm)", err_tbl, "{:.3f}"),
            ("Errors relative to the golden field", (rel_tbl * 100).round(2), "{:.2f} %"),
        ],
        notes=summary,
    )
    for fig in (fig_maps, fig_sc, fig_iwr):
        pdf.savefig(fig, bbox_inches="tight")
    _table_page(pdf, "IWR comparison", [("IWR per regime", iwr_tbl, "{:+.3f}")])

print(f"Saved → {pdf_path}")
