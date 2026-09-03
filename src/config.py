"""
LAFusion 2026: Spatio-Temporal Information Fusion Framework for Severe Wildfire Risk Prediction in the Brazilian Cerrado
Central Configuration Module (src/config.py)
"""

from pathlib import Path

# ==============================================================================
# 1. DIRECTORY PATHS (Automated Absolute Mapping)
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_RAW_INPE = DATA_RAW_DIR / "inpe_fires"
DATA_RAW_INMET = DATA_RAW_DIR / "inmet_weather"
DATA_RAW_MAPBIOMAS = DATA_RAW_DIR / "mapbiomas_landuse"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_FINAL_DIR = DATA_DIR / "final"

# Paper & Evaluation Assets Directories
PAPER_DIR = PROJECT_ROOT / "paper"
PAPER_TEMPLATE_DIR = PAPER_DIR / "template"
FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"

# Notebooks
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


# ==============================================================================
# 2. SPATIO-TEMPORAL SCOPE (10-Year Horizon: 2016-2025 | 12 Months)
# ==============================================================================
START_YEAR = 2016
END_YEAR = 2025
YEARS = list(range(START_YEAR, END_YEAR + 1))  # [2016, ..., 2025] (10 full years)
MONTHS = list(range(1, 13))                   # [1, ..., 12] (All 12 months)

# Geographic Focus: Goiás State & Federal District (Core Cerrado Biome)
TARGET_STATE = "GO"
TARGET_BIOME = "Cerrado"

# Bounding Box for Goiás / DF Coordinates
GEO_BOUNDS = {
    "lat_min": -19.50,
    "lat_max": -12.30,
    "lon_min": -53.30,
    "lon_max": -45.90,
}

# Grid Resolution for Spatio-Temporal Fusion
# ~0.1 degrees is approximately 11 km x 11 km at the equator/central Brazil
GRID_RESOLUTION_DEG = 0.10


# ==============================================================================
# 3. EXPERIMENTAL & MODELING PARAMETERS
# ==============================================================================
RANDOM_SEED = 42

# Target Variable and Severity Thresholds
TARGET_COLUMN = "severe_fire_risk_24h"   # Primary benchmark target: 24h ahead (t+1)
TARGET_COLUMN_48H = "severe_fire_risk_48h" # 48h ahead (t+2)
TARGET_COLUMN_72H = "severe_fire_risk_72h" # 72h ahead (t+3)
HORIZONS = [1, 2, 3]                     # Lead times in days (24h, 48h, 72h)

CRITICAL_FRP_THRESHOLD = 35.0       # Fire Radiative Power (MW) for severe classification
CRITICAL_HUMIDITY_THRESHOLD = 20.0  # Relative Humidity (%) for extreme dryness
CRITICAL_DAYS_WITHOUT_RAIN = 15     # Days without precipitation

# Cross-Validation Configuration (Temporal Split)
TRAIN_YEARS = list(range(2016, 2024))  # 2016-2023 (8 years training/historical)
TEST_YEARS = [2024, 2025]              # 2024-2025 (2 years out-of-time test set)

# Visuals & Figure Export Settings (Springer 300 DPI Standard)
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"  # Vectorial format for LaTeX / PNG for Word



# ==============================================================================
# 4. INITIALIZATION VERIFICATION HELPER
# ==============================================================================
def verify_environment():
    """Verify that all required directories exist."""
    required_dirs = [
        DATA_DIR, DATA_RAW_INPE, DATA_RAW_INMET, DATA_RAW_MAPBIOMAS,
        DATA_PROCESSED_DIR, DATA_FINAL_DIR, FIGURES_DIR, TABLES_DIR
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    return True


if __name__ == "__main__":
    verify_environment()
    print("=" * 70)
    print("LAFusion 2026 Central Configuration Loaded Successfully")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Temporal Scope: {START_YEAR} - {END_YEAR} ({len(YEARS)} years, 12 months/year)")
    print(f"Geographic Scope: {TARGET_STATE} ({TARGET_BIOME} Biome)")
    print(f"Grid Resolution: {GRID_RESOLUTION_DEG}° (~11 km)")
    print("=" * 70)
