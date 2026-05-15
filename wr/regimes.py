"""Regime metadata: index, short name, long name, color, and RGB values.

Index 0 = no regime. Indices 1-7 match the Cluster Class Index in
WR_LCattribution.txt (order: no AT ZO ScTr AR EuBL ScBL GL).
"""

REGIMES = [
    {"index": 0, "name": "no",   "long_name": "No regime",             "color": "grey",        "rgb": (128, 128, 128)},
    {"index": 1, "name": "AT",   "long_name": "Atlantic trough",       "color": "indigo",      "rgb": (75,  0,   130)},
    {"index": 2, "name": "ZO",   "long_name": "Zonal",                 "color": "red",         "rgb": (255, 0,     0)},
    {"index": 3, "name": "ScTr", "long_name": "Scandinavian trough",   "color": "darkorange",  "rgb": (255, 140,   0)},
    {"index": 4, "name": "AR",   "long_name": "Atlantic ridge",        "color": "gold",        "rgb": (255, 215,   0)},
    {"index": 5, "name": "EuBL", "long_name": "European blocking",     "color": "yellowgreen", "rgb": (154, 205,  50)},
    {"index": 6, "name": "ScBL", "long_name": "Scandinavian blocking", "color": "darkgreen",   "rgb": (0,   100,   0)},
    {"index": 7, "name": "GL",   "long_name": "Greenland blocking",    "color": "blue",        "rgb": (0,   0,   255)},
]

# Convenience lookups
BY_INDEX = {r["index"]: r for r in REGIMES}
BY_NAME  = {r["name"]:  r for r in REGIMES}

# Ordered names excluding "no", matching WRI_projections.txt column order
WR_NAMES = [r["name"] for r in REGIMES if r["name"] != "no"]
