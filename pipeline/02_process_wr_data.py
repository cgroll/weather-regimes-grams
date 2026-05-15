"""Process raw Grams weather regime text files into clean CSV files.

Produces:
  data/processed/wri_projections.csv   — datetime + WR index per regime
  data/processed/lc_attribution.csv    — datetime + EOF/max/lifecycle attribution
"""

import pandas as pd
from wr.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()


def parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y%m%d_%H")


# ---------------------------------------------------------------------------
# WRI_projections.txt
# ---------------------------------------------------------------------------
# Header lines (5):
#   1: description text
#   2: dashes
#   3: column description
#   4: dashes
#   5: blank

wri = pd.read_csv(
    paths.grams_wri_projections,
    skiprows=5,
    sep=r"\s+",
    header=None,
    names=["h_since", "datetime_str", "AT", "ZO", "ScTr", "AR", "EuBL", "ScBL", "GL"],
)
wri["datetime"] = parse_datetime(wri["datetime_str"])
wri = wri.drop(columns=["h_since", "datetime_str"])
wri = wri[["datetime", "AT", "ZO", "ScTr", "AR", "EuBL", "ScBL", "GL"]]

wri.to_csv(paths.wri_csv, index=False)
print(f"Saved {len(wri):,} rows → {paths.wri_csv}")
print(wri.head(3).to_string(index=False))


# ---------------------------------------------------------------------------
# WR_LCattribution.txt
# ---------------------------------------------------------------------------
# Header lines (7):
#   1: description text
#   2: dashes
#   3: cluster class index mapping
#   4: dashes
#   5: column description
#   6: dashes
#   7: blank

lc = pd.read_csv(
    paths.grams_lc_attribution,
    skiprows=7,
    sep=r"\s+",
    header=None,
    names=["h_since", "datetime_str", "eof_attribution", "max_wr_index", "lifecycle_wr_index"],
)
lc["datetime"] = parse_datetime(lc["datetime_str"])
lc = lc.drop(columns=["h_since", "datetime_str"])
lc = lc[["datetime", "eof_attribution", "max_wr_index", "lifecycle_wr_index"]]

lc.to_csv(paths.lc_attribution_csv, index=False)
print(f"\nSaved {len(lc):,} rows → {paths.lc_attribution_csv}")
print(lc.head(3).to_string(index=False))
