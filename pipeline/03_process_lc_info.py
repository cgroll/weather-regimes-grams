"""Process per-regime lifecycle information files into clean CSV files.

All 7 weather regime lifecycle files share the same 20-column schema and are
stacked into a single CSV with a leading `regime` column. The no-regime file
has a simpler 8-column schema and is stored separately.

Produces:
  data/processed/lc_info.csv       — stacked lifecycle events for 7 WRs
  data/processed/lc_no_regime.csv  — no-regime periods (onset/decay only)
"""

import pandas as pd
from wr.paths import ProjPaths
from wr.regimes import WR_NAMES

paths = ProjPaths()
paths.ensure_directories()

# ---------------------------------------------------------------------------
# 7 weather regime lifecycle files
# ---------------------------------------------------------------------------
# Header (11 lines to skip):
#   1-2: description
#   3:   dashes
#   4:   LIFECYCLE INFORMATION
#   5:   regime name
#   6:   clsfd EOF stats
#   7:   total mxI stats
#   8:   dashes
#   9:   column names
#   10:  dashes
#   11:  blank

WR_COLS = [
    "number",
    "onset", "sat_start", "mx", "sat_end", "decay",
    "dcfr", "dcto", "dctoID", "dctoDATE",
    "onfr", "onto", "onfrID", "onfromDATE",
    "trfr", "trfrID", "trfromDATE",
    "trto", "trtoID", "trtoDATE",
]
WR_DATE_COLS = ["onset", "sat_start", "mx", "sat_end", "decay",
                "dctoDATE", "onfromDATE", "trfromDATE", "trtoDATE"]
WR_ID_COLS   = ["dctoID", "onfrID", "trfrID", "trtoID"]

dfs = []
for name in WR_NAMES:
    path = paths.grams_wr_data_path / f"WR_lifecycle_information_{name}.txt"
    df = pd.read_csv(path, skiprows=11, sep=r"\s+", header=None, names=WR_COLS)
    df.insert(0, "regime", name)
    dfs.append(df)
    print(f"  {name}: {len(df)} life cycles")

lc = pd.concat(dfs, ignore_index=True)

for col in WR_DATE_COLS:
    lc[col] = pd.to_datetime(lc[col], format="%Y%m%d_%H", errors="coerce")

for col in WR_ID_COLS:
    lc[col] = lc[col].replace(-999, pd.NA)

lc.to_csv(paths.lc_info_csv, index=False)
print(f"\nSaved {len(lc):,} rows → {paths.lc_info_csv}")
print(lc.head(3).to_string(index=False))

# ---------------------------------------------------------------------------
# No-regime lifecycle file (different schema)
# ---------------------------------------------------------------------------
# Header (11 lines to skip — same count, but lines 6-7 are blank instead of
# stats, then dashes / column description / dashes / blank)

NO_COLS = ["number", "onset", "decay", "duration", "comes_from", "id_from",
           "transition_to", "id_to"]

no_path = paths.grams_wr_data_path / "WR_lifecycle_information_no.txt"
lc_no = pd.read_csv(no_path, skiprows=11, sep=r"\s+", header=None, names=NO_COLS)

for col in ["onset", "decay"]:
    lc_no[col] = pd.to_datetime(lc_no[col], format="%Y%m%d_%H")

for col in ["id_from", "id_to"]:
    lc_no[col] = lc_no[col].replace(-999, pd.NA)

lc_no.to_csv(paths.lc_no_regime_csv, index=False)
print(f"\nSaved {len(lc_no):,} rows → {paths.lc_no_regime_csv}")
print(lc_no.head(3).to_string(index=False))
