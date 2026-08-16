# A Spatio-Temporal Information Fusion Framework for Severe Wildfire Risk Prediction in the Brazilian Cerrado

[![Event](https://img.shields.io/badge/Conference-LAFusion%202026-blue.svg)](https://lafusion.cos.ufrj.br/)
[![Proceedings](https://img.shields.io/badge/Publication-Springer%20CCIS-orange.svg)](https://www.springer.com/series/7899)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic%20Research-lightgrey.svg)](#)

---

## 📌 1. Overview & Research Scope

This repository hosts the official data engineering pipeline, machine learning models, and evaluation scripts for the scientific paper submitted to the **Fourth Latin American Workshop on Information Fusion (LAFusion 2026)**, to be published in the **Springer Communications in Computer and Information Science (CCIS)** series.

### 🎯 Research Objective
Wildfires in the Brazilian Cerrado pose severe ecological, economic, and public health threats. Existing early-warning systems are either **purely reactive** (detecting fire thermal anomalies only after ignition via satellite) or rely on **isolated meteorological models** that produce high false-positive rates by ignoring land use, fuel biomass, and historical vulnerabilities.

This research introduces a **Spatio-Temporal Information Fusion Framework** that integrates three heterogeneous public data streams to provide proactive (24h–72h) wildfire risk forecasts across a **10-year period (2016–2025)** covering all **12 months** of the year.

---

## 🧩 2. Heterogeneous Data Streams & Information Fusion Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │ 1. REMOTE SENSING & THERMAL ANOMALIES (INPE / NASA)    │
                  │ - Active fire detections, coordinates, FRP (MW)        │
                  │ - Historical frequency & fire severity patterns        │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                  ┌──────────────────────────▼─────────────────────────────┐
                  │ 2. DYNAMIC METEOROLOGY (INMET Surface Weather Stations)│
                  │ - Maximum/average daily temperature, relative humidity │
                  │ - Wind speed and gusts, consecutive dry days (DSR)     │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                  ┌──────────────────────────▼─────────────────────────────┐
                  │ 3. LAND USE & VEGETATION FUEL (MapBiomas Brasil)       │
                  │ - Land cover class (pasture, savanna, crop, forest)    │
                  │ - Proximity to agricultural borders & roads            │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                                  ┌──────────▼──────────┐
                                  │ SPATIO-TEMPORAL     │
                                  │ GRID ALIGNMENT      │
                                  │ (0.10° ~ 11 km)     │
                                  └──────────┬──────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     │                                               │
          ┌──────────▼──────────┐                         ┌──────────▼──────────┐
          │ EARLY FEATURE       │                         │ LATE DECISION       │
          │ FUSION              │                         │ FUSION              │
          │ (XGBoost / RF /     │                         │ (Dempster-Shafer /  │
          │  LightGBM)          │                         │  Weighted Soft-Vote)│
          └──────────┬──────────┘                         └──────────┬──────────┘
                     │                                               │
                     └───────────────────────┬───────────────────────┘
                                             │
                                  ┌──────────▼──────────┐
                                  │ EVALUATION & SHAP   │
                                  │ EXPLAINABILITY      │
                                  │ (300 DPI Figures)   │
                                  └─────────────────────┘
```

---

## 🗂️ 3. Project Directory Structure

```text
LaFusion/
├── data/
│   ├── raw/                  # Raw downloaded datasets
│   │   ├── inpe_fires/       # INPE BDQueimadas / NASA FIRMS CSVs
│   │   ├── inmet_weather/    # INMET hourly/daily meteorological series
│   │   └── mapbiomas_landuse/# MapBiomas land cover rasters and tables
│   ├── processed/            # Cleaned, standardized intermediate data
│   └── final/                # Unified spatio-temporal fusion matrix
├── src/
│   ├── config.py             # Central configuration (10-year scope, bounds, seeds)
│   ├── ingestion/            # Automated extraction and download pipelines
│   ├── preprocessing/        # Data cleansing, spatial grid mapping & normalization
│   ├── fusion/               # Baseline, Early Fusion and Late Fusion models
│   └── evaluation/           # Temporal cross-validation, metrics & SHAP explainability
├── notebooks/                # Exploratory analysis & visual prototyping
├── paper/
│   ├── template/             # Springer LNCS / CCIS template files
│   ├── figures/              # Publication-ready vector graphics (300 DPI)
│   └── tables/               # Formatted metric tables for the manuscript
├── Historico_e_Planejamento_LAFusion_2026.md
├── Estrutura_e_Fases_do_Projeto.md
├── LAFusion_2026_Informacoes.md
├── requirements.txt          # Production dependencies
└── README.md                 # Technical project documentation
```

---

## ⚡ 4. Quick Start & Setup

### Prerequisites
* Python 3.10+
* Virtual environment (recommended)

### Installation
```bash
# 1. Clone or navigate to the repository directory
cd "C:\Users\joaos\OneDrive\Documentos\2026\01_Academico_e_UFG\Trabalhos\LaFusion"

# 2. Create and activate a Python virtual environment (optional)
python -m venv venv
.\venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt
```

### Verifying Environment Configuration
```bash
python src/config.py
```

---

## 🗓️ 5. Research Phases & Roadmap

| Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 1** | Planning, Software Architecture, Directory Setup & Configuration | **Completed** |
| **Phase 2** | Data Engineering: Ingestion (INPE, INMET, MapBiomas) & Grid Alignment | **Next Step** |
| **Phase 3** | Model Development: Baselines, Early Fusion & Late (Dempster-Shafer) Fusion | Pending |
| **Phase 4** | Scientific Experimentation: Temporal CV, Metrics & SHAP Explainability | Pending |
| **Phase 5** | Technical Paper Authoring (Springer CCIS 12-page format in English) | Pending |
| **Phase 6** | Quality Assurance (Double-Blind Audit) & Microsoft CMT Submission | Pending (Deadline: 04/09/2026) |

---

## 📜 6. Authorship & Double-Blind Review Policy
In compliance with the **Springer CCIS / LAFusion 2026 Double-Blind Review** policy, this repository and all manuscript drafts are anonymized during the peer-review cycle.
