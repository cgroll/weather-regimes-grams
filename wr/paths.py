"""Project paths configuration.

All paths are resolved relative to the project root, making scripts runnable
from any working directory. Add a @property for each new data file introduced
in the pipeline.
"""

from pathlib import Path


class ProjPaths:
    """Centralized project paths.

    The root is inferred from the location of this file (pkg/), so scripts
    run correctly regardless of the working directory they are invoked from.
    """

    def __init__(self):
        self._pkg_path = Path(__file__).resolve().parent  # pkg/
        self._project_path = self._pkg_path.parent        # project root

    # ------------------------------------------------------------------ #
    # Top-level directories                                                #
    # ------------------------------------------------------------------ #

    @property
    def project_path(self) -> Path:
        """Root project directory."""
        return self._project_path

    @property
    def pkg_path(self) -> Path:
        """Source package directory (pkg/)."""
        return self._pkg_path

    @property
    def pipeline_path(self) -> Path:
        """Pipeline scripts directory."""
        return self._project_path / "pipeline"

    # ------------------------------------------------------------------ #
    # Data directories                                                     #
    # ------------------------------------------------------------------ #

    @property
    def data_path(self) -> Path:
        """Main data directory."""
        return self._project_path / "data"

    @property
    def downloads_path(self) -> Path:
        """Raw downloaded data."""
        return self.data_path / "downloads"

    @property
    def processed_data_path(self) -> Path:
        """Processed/transformed data."""
        return self.data_path / "processed"

    # ------------------------------------------------------------------ #
    # Output directories                                                   #
    # ------------------------------------------------------------------ #

    @property
    def output_path(self) -> Path:
        """Generated outputs root."""
        return self._project_path / "output"

    @property
    def images_path(self) -> Path:
        """Chart/figure images saved by pipeline scripts."""
        return self.output_path / "images"

    @property
    def reports_path(self) -> Path:
        """Report files."""
        return self.output_path / "reports"

    # ------------------------------------------------------------------ #
    # Grams weather regimes dataset (Zenodo 17080146)                     #
    # ------------------------------------------------------------------ #

    @property
    def grams_data_path(self) -> Path:
        """Root of the extracted Zenodo data package."""
        return self.downloads_path / "wr_data_package_V1.0"

    @property
    def grams_wr_data_path(self) -> Path:
        """Core regime data files (NetCDF + text)."""
        return self.grams_data_path / "wr_data"

    @property
    def grams_clusters(self) -> Path:
        """Cluster centroids NetCDF."""
        return self.grams_wr_data_path / "Clusters_WRs.nc"

    @property
    def grams_eofs(self) -> Path:
        """EOF patterns NetCDF."""
        return self.grams_wr_data_path / "EOFs_WRs.nc"

    @property
    def grams_z500_patterns(self) -> Path:
        """Normalised Z500 patterns on the EOF domain."""
        return self.grams_wr_data_path / "Normed_Z0500-patterns_EOFdomain.nc"

    @property
    def grams_wri_projections(self) -> Path:
        """Weather regime index projections (full time series)."""
        return self.grams_wr_data_path / "WRI_projections.txt"

    @property
    def grams_lc_attribution(self) -> Path:
        """Life-cycle attribution table."""
        return self.grams_wr_data_path / "WR_LCattribution.txt"

    @property
    def grams_std_params(self) -> Path:
        """Standardisation parameters (mean + std of projections, 1979-2019)."""
        return self.grams_wr_data_path / "WRI_std_params.txt"

    @property
    def grams_example_data_path(self) -> Path:
        """Example data directory (one Z500 field + climatology)."""
        return self.grams_data_path / "example_data"

    @property
    def grams_example_z500(self) -> Path:
        """Example 10-day LP-filtered Z500 anomaly field (2025-06-01 00 UTC)."""
        return self.grams_example_data_path / "Z0500_20250601_00.nc"

    @property
    def grams_clim_z500(self) -> Path:
        """Year-round 1979-2019 climatological mean of Z500 (geopotential, m²/s²)."""
        return self.grams_example_data_path / "CLIM_Z@500_year_1979-2019.nc"

    # ------------------------------------------------------------------ #
    # ERA5 data                                                            #
    # ------------------------------------------------------------------ #

    @property
    def era5_path(self) -> Path:
        """Root directory for downloaded ERA5 fields."""
        return self.downloads_path / "era5"

    def era5_z500_nc(self, start: str, end: str) -> Path:
        """NetCDF file for ERA5 Z0500 covering a given analysis period.

        start / end: 'YYYY-MM-DD' strings for the *analysis* period (without
        the filter padding — padding is embedded in the file content but the
        filename reflects the analysis window the user requested).
        """
        s = start.replace("-", "")
        e = end.replace("-", "")
        return self.era5_path / f"z0500_{s}_{e}.nc"

    # ------------------------------------------------------------------ #
    # Processed data files                                                 #
    # ------------------------------------------------------------------ #

    @property
    def wri_csv(self) -> Path:
        """Processed WR index projections (datetime + one column per regime)."""
        return self.processed_data_path / "wri_projections.csv"

    @property
    def lc_attribution_csv(self) -> Path:
        """Processed life-cycle attribution (datetime + eof/max/lifecycle columns)."""
        return self.processed_data_path / "lc_attribution.csv"

    @property
    def lc_info_csv(self) -> Path:
        """Stacked lifecycle events for the 7 weather regimes."""
        return self.processed_data_path / "lc_info.csv"

    @property
    def lc_no_regime_csv(self) -> Path:
        """No-regime periods (onset/decay/duration/transition)."""
        return self.processed_data_path / "lc_no_regime.csv"

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def ensure_directories(self) -> None:
        """Create all standard directories if they do not yet exist."""
        dirs = [
            self.downloads_path,
            self.processed_data_path,
            self.images_path,
            self.reports_path,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
