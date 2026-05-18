# Weather Regimes — Grams

Analysis and replication of [Christian Grams'](https://www.imk-tro.kit.edu/english/staff_grams.php)
weather regime classification for Europe/Atlantic.
The dataset is published on Zenodo:
[record 17080146](https://zenodo.org/records/17080146).

The data pipeline is managed by [Snakemake](https://snakemake.readthedocs.io/);
dependencies are managed by [uv](https://docs.astral.sh/uv/).
Results are published as a [MyST](https://mystmd.org/) Jupyter Book on GitHub Pages.

## Quick start

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync

make dry-run   # preview what would run
make run       # download data and execute the pipeline
make serve     # open http://localhost:3000 — live book preview
```

## Data

The raw dataset is downloaded automatically on the first `make run`.
It is fetched from:

```
https://zenodo.org/api/records/17080146/files-archive
```

The archive is saved to `data/raw/grams_weather_regimes.zip` and extracted
in-place. Both the zip and extracted files are git-ignored; re-running
`snakemake` will not re-download unless the zip is deleted.

To force a fresh download:

```bash
rm data/raw/grams_weather_regimes.zip
make run
```

## Project layout

```
project-root/
├── wr/                  # Python package — shared utilities
│   └── paths.py         # Centralized path config
├── pipeline/            # Pipeline scripts
│   ├── 01_download_*    # Data acquisition (no charts)
│   └── 02_analyse_*     # Analysis → notebook
├── book/                # MyST book source
│   ├── notebooks/       # Executed notebooks (Snakemake output)
│   ├── markdown/        # Static content
│   └── myst.yml         # TOC and site settings
├── data/
│   ├── raw/             # Downloaded Zenodo archive + extracted files
│   └── processed/       # Derived datasets (both git-ignored)
├── output/images/       # Figures (tracked in git)
├── Snakefile            # Pipeline DAG
└── contribution_conventions.md   # Conventions for contributors/AI
```

See [contribution_conventions.md](contribution_conventions.md) for full details on
adding pipeline stages, writing analysis scripts, and Snakemake usage.

## Common Snakemake commands

| Command | Effect |
|---------|--------|
| `snakemake -n` | Dry run — show what would execute |
| `snakemake -j4` | Run pipeline (4 parallel jobs) |
| `snakemake -R <rule>` | Force-re-run a specific rule |
| `snakemake <file>` | Build one specific output file |
| `snakemake --forceall` | Re-run everything unconditionally |

## GitHub Pages

In your repository: **Settings → Pages → Source → GitHub Actions**.
Every push to `main` builds and deploys the book automatically.
Pull requests run only the build check.

## Lifecycle attribution algorithm

The pre-computed `WR_LCattribution.txt` column 5 (`lifecycle_wr_index`) is the
primary attribution used for most analyses.  `pipeline/05_compute_projection.py`
reproduces it with the following empirically inferred rules:

### Rules (inferred from Grams dataset, 2025-05-17)

| Parameter | Value | Notes |
|-----------|-------|-------|
| IWR threshold | ≥ 1.0 | exact: onset IWR = 1.0, step before < 1.0; decay IWR = 1.0, step after < 1.0 |
| Minimum lifecycle duration | 40 timesteps = 120 h = **5 days** | shortest observed lifecycle in dataset is exactly 40 steps |
| Max bridgeable gap | 40 timesteps = 120 h = **5 days** | IWR may dip below 1.0 for ≤ 40 steps without terminating the lifecycle; max observed dip within a lifecycle is exactly 40 steps |
| Overlap | allowed | multiple regimes can have active lifecycles simultaneously |
| Dominant regime | highest IWR wins | when lifecycles overlap, the regime with the largest IWR at that timestep is reported |

### Validation

| Metric | Result |
|--------|--------|
| Lifecycle-level match (onset/decay pair exact) | **2097 / 2136 = 98.2 %** |
| Timestep attribution match | **213 294 / 220 728 = 96.6 %** |

The gap between the two metrics is caused by over-detection: the algorithm finds
2655 lifecycle segments vs 2136 in the Grams dataset.  The extra 519 segments
create spurious active-regime periods in the timestep attribution.

### TODO — close the remaining 1.8 % (39 misses)

The 39 unmatched lifecycles split into two failure modes:

1. **Algorithm merges, Grams splits** — two Grams lifecycles separated by a gap
   of ≤ 40 steps are merged into one by the bridge rule (e.g. AT
   1985-07-31→08-27 and AT 1985-08-30→09-05, separated by only 9 steps).
   Hypothesis: Grams forces a split when a *different* regime's lifecycle becomes
   dominant during the gap, even if the original regime's IWR never drops below
   1.0 for long enough to trigger a decay on its own.

2. **Boundary shift** — onset or decay differs by a few timesteps (algorithm
   traces a longer run back to a slightly earlier crossing).

To investigate: for each miss, check whether a different regime is dominant
during the bridged gap, and test a refined rule such as
*"split a lifecycle whenever a different regime's lifecycle is dominant for the
entire gap period."*  Aim: reproduce all 2136 lifecycles exactly and eliminate
dependence on the pre-computed `WR_lifecycle_information_*.txt` files.
