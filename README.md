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
