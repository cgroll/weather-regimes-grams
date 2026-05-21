# Snakefile — weather-regimes-grams pipeline
#
#   ancient(path)   — treat the file as infinitely old if it exists; skips
#                     the rule without re-downloading. Delete the file to
#                     force a fresh fetch.
#
#   rule all        — pseudo-rule listing the final targets that bare
#                     `snakemake` will build.
#
# Common commands:
#   snakemake -n                  dry-run
#   snakemake -j4                 run with 4 parallel jobs
#   snakemake <target>            build one specific output file
#   snakemake --forcerun <rule>   force a rule to re-run

# ---------------------------------------------------------------------------
# Project-wide settings
# ---------------------------------------------------------------------------

ANALYSIS_NOTEBOOKS = [
    "book/notebooks/04_wr_timeseries.ipynb",
    "book/notebooks/05_compute_projection.ipynb",
    "book/notebooks/06_era5_wr_projection.ipynb",
    "book/notebooks/07_wr_analogs.ipynb",
    "book/notebooks/08_lifecycle_inspection.ipynb",
    "book/notebooks/09_pecd_overview.ipynb",
    "book/notebooks/10_wr_pecd_germany.ipynb",
    "book/notebooks/11_wr_pecd_maps.ipynb",
    "book/notebooks/12_low_cf_events.ipynb",
    "book/notebooks/13_low_wind_regimes.ipynb",
    "book/notebooks/14_pecd_de_climatology.ipynb",
]

PROCESSED_DATA = [
    "data/processed/wri_projections.csv",
    "data/processed/lc_attribution.csv",
    "data/processed/lc_info.csv",
    "data/processed/lc_no_regime.csv",
]

ERA5_YEARS = [2019, 2020, 2021]        # expand to list(range(1960, 2022)) for the full dataset

DOWNLOADS = [
    "data/downloads/wb/z500_climatology.zarr",
    "data/downloads/era5/z500_euro_atlantic.zarr",
]

GENERATED_IMAGES = [
    "output/images/18_z500_climatology.gif",
] + expand("output/images/19_z500_{year}.gif", year=ERA5_YEARS) \
  + expand("output/images/19_z500_anomaly_{year}.gif", year=ERA5_YEARS)

# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------

rule all:
    input:
        PROCESSED_DATA + ANALYSIS_NOTEBOOKS + DOWNLOADS + GENERATED_IMAGES

# ---------------------------------------------------------------------------
# Download rules
# ---------------------------------------------------------------------------

rule download_era5_z500:
    output:
        "data/downloads/era5/z0500_20241101_20250331.nc",
    shell:
        "uv run python pipeline/01_download_era5_z500.py"

rule download_grams:
    output:
        "data/downloads/wr_data_package_V1.0/wr_data/Clusters_WRs.nc",
    shell:
        "uv run python pipeline/01_download_grams.py"

rule download_wb_z500_climatology:
    output:
        directory("data/downloads/wb/z500_climatology.zarr"),
    shell:
        "uv run python pipeline/15_download_wb_z500_climatology.py"

rule download_era5_z500_year:
    output:
        directory("data/downloads/era5/z500_years/{year}.zarr"),
    wildcard_constraints:
        year = r"\d{4}",
    shell:
        "uv run python pipeline/16_download_era5_z500_daily.py {wildcards.year}"

rule concat_era5_z500:
    input:
        expand("data/downloads/era5/z500_years/{year}.zarr", year=ERA5_YEARS),
    output:
        directory("data/downloads/era5/z500_euro_atlantic.zarr"),
    shell:
        "uv run python pipeline/17_concat_era5_z500.py"

# ---------------------------------------------------------------------------
# Image generation rules
# ---------------------------------------------------------------------------

rule z500_climatology_gif:
    input:
        script = "pipeline/18_z500_climatology_gif.py",
        clim   = "data/downloads/wb/z500_climatology.zarr",
    output:
        "output/images/18_z500_climatology.gif",
    shell:
        "uv run python pipeline/18_z500_climatology_gif.py"

rule z500_anomaly_gif:
    input:
        script = "pipeline/19_z500_anomaly_gif.py",
        era5   = "data/downloads/era5/z500_years/{year}.zarr",
        clim   = "data/downloads/wb/z500_climatology.zarr",
    output:
        raw  = "output/images/19_z500_{year}.gif",
        anom = "output/images/19_z500_anomaly_{year}.gif",
    wildcard_constraints:
        year = r"\d{4}",
    shell:
        "uv run python pipeline/19_z500_anomaly_gif.py {wildcards.year}"

# ---------------------------------------------------------------------------
# Reference material
# ---------------------------------------------------------------------------

rule convert_example_notebook:
    input:
        "data/downloads/wr_data_package_V1.0/scripts_first_steps/WR_read_example.ipynb",
    output:
        "data/downloads/wr_data_package_V1.0/scripts_first_steps/WR_read_example.py",
    shell:
        """
        uv run jupyter nbconvert --to script {input} \
            --output WR_read_example \
            --output-dir data/downloads/wr_data_package_V1.0/scripts_first_steps/
        """

# ---------------------------------------------------------------------------
# Processing rules
# ---------------------------------------------------------------------------

rule process_wr_data:
    input:
        wri = "data/downloads/wr_data_package_V1.0/wr_data/WRI_projections.txt",
        lc  = "data/downloads/wr_data_package_V1.0/wr_data/WR_LCattribution.txt",
        script = "pipeline/02_process_wr_data.py",
    output:
        wri_csv = "data/processed/wri_projections.csv",
        lc_csv  = "data/processed/lc_attribution.csv",
    shell:
        "uv run python pipeline/02_process_wr_data.py"

rule process_lc_info:
    input:
        lc_files = expand(
            "data/downloads/wr_data_package_V1.0/wr_data/WR_lifecycle_information_{regime}.txt",
            regime=["AT", "ZO", "ScTr", "AR", "EuBL", "ScBL", "GL", "no"],
        ),
        script = "pipeline/03_process_lc_info.py",
    output:
        lc_info    = "data/processed/lc_info.csv",
        lc_no      = "data/processed/lc_no_regime.csv",
    shell:
        "uv run python pipeline/03_process_lc_info.py"

# ---------------------------------------------------------------------------
# Analysis / notebook rules
# ---------------------------------------------------------------------------

rule era5_wr_projection:
    input:
        script   = "pipeline/06_era5_wr_projection.py",
        z500     = "data/downloads/era5/z0500_20241101_20250331.nc",
        clim     = "data/downloads/wr_data_package_V1.0/example_data/CLIM_Z@500_year_1979-2019.nc",
        eofs     = "data/downloads/wr_data_package_V1.0/wr_data/EOFs_WRs.nc",
        patterns = "data/downloads/wr_data_package_V1.0/wr_data/Normed_Z0500-patterns_EOFdomain.nc",
        wri_csv  = "data/processed/wri_projections.csv",
    output:
        notebook = "book/notebooks/06_era5_wr_projection.ipynb",
        img      = "output/images/06_era5_iwr_comparison.png",
    shell:
        """
        MPLBACKEND=Agg uv run jupytext --to notebook --execute \
            --set-kernel python3 \
            --output {output.notebook} {input.script} && \
        uv run python -c "
import nbformat
nb = nbformat.read('{output.notebook}', as_version=4)
nb.cells = [c for c in nb.cells
            if not (c.cell_type == 'raw' and 'jupytext' in c.source)]
nb.metadata.pop('jupytext', None)
nbformat.write(nb, '{output.notebook}')
"
        """

rule compute_projection:
    input:
        script   = "pipeline/05_compute_projection.py",
        z500     = "data/downloads/wr_data_package_V1.0/example_data/Z0500_20250601_00.nc",
        eofs     = "data/downloads/wr_data_package_V1.0/wr_data/EOFs_WRs.nc",
        patterns = "data/downloads/wr_data_package_V1.0/wr_data/Normed_Z0500-patterns_EOFdomain.nc",
        wri_csv  = "data/processed/wri_projections.csv",
        lc_csv   = "data/processed/lc_attribution.csv",
        lc_info  = "data/processed/lc_info.csv",
    output:
        notebook = "book/notebooks/05_compute_projection.ipynb",
    shell:
        """
        MPLBACKEND=Agg uv run jupytext --to notebook --execute \
            --set-kernel python3 \
            --output {output.notebook} {input.script} && \
        uv run python -c "
import nbformat
nb = nbformat.read('{output.notebook}', as_version=4)
nb.cells = [c for c in nb.cells
            if not (c.cell_type == 'raw' and 'jupytext' in c.source)]
nb.metadata.pop('jupytext', None)
nbformat.write(nb, '{output.notebook}')
"
        """

rule lifecycle_inspection:
    input:
        script  = "pipeline/08_lifecycle_inspection.py",
        lc_info = "data/processed/lc_info.csv",
        lc_no   = "data/processed/lc_no_regime.csv",
        lc_csv  = "data/processed/lc_attribution.csv",
    output:
        notebook   = "book/notebooks/08_lifecycle_inspection.ipynb",
        img_dur       = "output/images/08_lc_duration_jitter.png",
        img_trans     = "output/images/08_transition_matrix.png",
        img_trans_all = "output/images/08_transition_matrix_all.png",
    shell:
        """
        MPLBACKEND=Agg uv run jupytext --to notebook --execute \
            --set-kernel python3 \
            --output {output.notebook} {input.script} && \
        uv run python -c "
import nbformat
nb = nbformat.read('{output.notebook}', as_version=4)
nb.cells = [c for c in nb.cells
            if not (c.cell_type == 'raw' and 'jupytext' in c.source)]
nb.metadata.pop('jupytext', None)
nbformat.write(nb, '{output.notebook}')
"
        """

rule wr_analogs:
    input:
        script   = "pipeline/07_wr_analogs.py",
        wri_csv  = "data/processed/wri_projections.csv",
        lc_info  = "data/processed/lc_info.csv",
        lc_csv   = "data/processed/lc_attribution.csv",
    output:
        notebook = "book/notebooks/07_wr_analogs.ipynb",
        img_jitter   = "output/images/07_analog_jitter.png",
        img_diverg   = "output/images/07_analog_divergence.png",
        img_dist_all = "output/images/07_pairwise_dist.png",
        img_dist_reg = "output/images/07_pairwise_dist_regimes.png",
        img_heatmap      = "output/images/07_analog_regime_heatmap.png",
        img_fracs        = "output/images/07_analog_regime_fracs.png",
        img_heatmap_back = "output/images/07_analog_regime_heatmap_back.png",
        img_fracs_back   = "output/images/07_analog_regime_fracs_back.png",
    shell:
        """
        MPLBACKEND=Agg uv run jupytext --to notebook --execute \
            --set-kernel python3 \
            --output {output.notebook} {input.script} && \
        uv run python -c "
import nbformat
nb = nbformat.read('{output.notebook}', as_version=4)
nb.cells = [c for c in nb.cells
            if not (c.cell_type == 'raw' and 'jupytext' in c.source)]
nb.metadata.pop('jupytext', None)
nbformat.write(nb, '{output.notebook}')
"
        """


rule pecd_de_climatology:
    input:
        script = "pipeline/14_pecd_de_climatology.py",
        pecd   = "/home/chris/research/world-of-energy/data/processed/pecd/pecd_regions.parquet",
    output:
        notebook = "book/notebooks/14_pecd_de_climatology.ipynb",
        img      = "output/images/14_pecd_de_climatology.png",
    shell:
        """
        MPLBACKEND=Agg uv run jupytext --to notebook --execute \
            --set-kernel python3 \
            --output {output.notebook} {input.script} && \
        uv run python -c "
import nbformat
nb = nbformat.read('{output.notebook}', as_version=4)
nb.cells = [c for c in nb.cells
            if not (c.cell_type == 'raw' and 'jupytext' in c.source)]
nb.metadata.pop('jupytext', None)
nbformat.write(nb, '{output.notebook}')
"
        """

rule low_wind_regimes:
    input:
        script  = "pipeline/13_low_wind_regimes.py",
        pecd    = "/home/chris/research/world-of-energy/data/processed/pecd/pecd_regions.parquet",
        lc_csv  = "data/processed/lc_attribution.csv",
    output:
        notebook  = "book/notebooks/13_low_wind_regimes.ipynb",
        img_simul      = "output/images/13_low_wind_regime_stacks.png",
        img_wind_lags  = expand("output/images/13_low_wind_lag_{lag}d.png",
                                lag=["05", "10", "15", "20", "25"]),
        img_solar_simul = "output/images/13_solar_regime_stacks.png",
        img_solar_lags  = expand("output/images/13_solar_lag_{lag}d.png",
                                 lag=["05", "10", "15", "20", "25"]),
    shell:
        """
        MPLBACKEND=Agg uv run jupytext --to notebook --execute \
            --set-kernel python3 \
            --output {output.notebook} {input.script} && \
        uv run python -c "
import nbformat
nb = nbformat.read('{output.notebook}', as_version=4)
nb.cells = [c for c in nb.cells
            if not (c.cell_type == 'raw' and 'jupytext' in c.source)]
nb.metadata.pop('jupytext', None)
nbformat.write(nb, '{output.notebook}')
"
        """

rule low_cf_events:
    input:
        script   = "pipeline/12_low_cf_events.py",
        pecd     = "/home/chris/research/world-of-energy/data/processed/pecd/pecd_regions.parquet",
        wri_csv  = "data/processed/wri_projections.csv",
        lc_csv   = "data/processed/lc_attribution.csv",
    output:
        notebook       = "book/notebooks/12_low_cf_events.ipynb",
        img_solar      = "output/images/12_low_cf_worst_solar.png",
        img_wind_on    = "output/images/12_low_cf_worst_wind_on.png",
        img_wind_off   = "output/images/12_low_cf_worst_wind_off.png",
    shell:
        """
        MPLBACKEND=Agg uv run jupytext --to notebook --execute \
            --set-kernel python3 \
            --output {output.notebook} {input.script} && \
        uv run python -c "
import nbformat
nb = nbformat.read('{output.notebook}', as_version=4)
nb.cells = [c for c in nb.cells
            if not (c.cell_type == 'raw' and 'jupytext' in c.source)]
nb.metadata.pop('jupytext', None)
nbformat.write(nb, '{output.notebook}')
"
        """

rule wr_pecd_maps:
    input:
        script  = "pipeline/11_wr_pecd_maps.py",
        pecd    = "/home/chris/research/world-of-energy/data/processed/pecd/pecd_regions.parquet",
        lc_csv  = "data/processed/lc_attribution.csv",
    output:
        notebook = "book/notebooks/11_wr_pecd_maps.ipynb",
        imgs     = expand(
            "output/images/11_wr_maps_{regime}.png",
            regime=["no", "AT", "ZO", "ScTr", "AR", "EuBL", "ScBL", "GL"],
        ),
    shell:
        """
        MPLBACKEND=Agg uv run jupytext --to notebook --execute \
            --set-kernel python3 \
            --output {output.notebook} {input.script} && \
        uv run python -c "
import nbformat
nb = nbformat.read('{output.notebook}', as_version=4)
nb.cells = [c for c in nb.cells
            if not (c.cell_type == 'raw' and 'jupytext' in c.source)]
nb.metadata.pop('jupytext', None)
nbformat.write(nb, '{output.notebook}')
"
        """

rule wr_pecd_germany:
    input:
        script   = "pipeline/10_wr_pecd_germany.py",
        pecd     = "/home/chris/research/world-of-energy/data/processed/pecd/pecd_regions.parquet",
        wri_csv  = "data/processed/wri_projections.csv",
        lc_csv   = "data/processed/lc_attribution.csv",
    output:
        notebook  = "book/notebooks/10_wr_pecd_germany.ipynb",
        img_scat    = "output/images/10_wr_de_scatter.png",
        img_box     = "output/images/10_wr_de_boxplot.png",
        img_corr    = "output/images/10_wr_de_corr.png",
        img_box_met = "output/images/10_wr_de_boxplot_met.png",
    shell:
        """
        MPLBACKEND=Agg uv run jupytext --to notebook --execute \
            --set-kernel python3 \
            --output {output.notebook} {input.script} && \
        uv run python -c "
import nbformat
nb = nbformat.read('{output.notebook}', as_version=4)
nb.cells = [c for c in nb.cells
            if not (c.cell_type == 'raw' and 'jupytext' in c.source)]
nb.metadata.pop('jupytext', None)
nbformat.write(nb, '{output.notebook}')
"
        """

rule pecd_overview:
    input:
        script = "pipeline/09_pecd_overview.py",
        data   = "/home/chris/research/world-of-energy/data/processed/pecd/pecd_regions.parquet",
    output:
        notebook      = "book/notebooks/09_pecd_overview.ipynb",
        img_tseries   = "output/images/09_pecd_timeseries.png",
        img_seasonal  = "output/images/09_pecd_seasonal.png",
        img_countries = "output/images/09_pecd_country_comparison.png",
    shell:
        """
        MPLBACKEND=Agg uv run jupytext --to notebook --execute \
            --set-kernel python3 \
            --output {output.notebook} {input.script} && \
        uv run python -c "
import nbformat
nb = nbformat.read('{output.notebook}', as_version=4)
nb.cells = [c for c in nb.cells
            if not (c.cell_type == 'raw' and 'jupytext' in c.source)]
nb.metadata.pop('jupytext', None)
nbformat.write(nb, '{output.notebook}')
"
        """

rule wr_timeseries:
    input:
        script   = "pipeline/04_wr_timeseries.py",
        wri_csv  = "data/processed/wri_projections.csv",
        lc_csv   = "data/processed/lc_info.csv",
        attr_csv = "data/processed/lc_attribution.csv",
    output:
        notebook   = "book/notebooks/04_wr_timeseries.ipynb",
        img_tseries = "output/images/04_wr_timeseries.png",
        img_freq    = "output/images/04_wr_freq_overall.png",
        img_annual  = "output/images/04_wr_freq_annual.png",
        img_cal     = "output/images/04_wr_calendar.png",
    shell:
        """
        MPLBACKEND=Agg uv run jupytext --to notebook --execute \
            --set-kernel python3 \
            --output {output.notebook} {input.script} && \
        uv run python -c "
import nbformat
nb = nbformat.read('{output.notebook}', as_version=4)
nb.cells = [c for c in nb.cells
            if not (c.cell_type == 'raw' and 'jupytext' in c.source)]
nb.metadata.pop('jupytext', None)
nbformat.write(nb, '{output.notebook}')
"
        """
