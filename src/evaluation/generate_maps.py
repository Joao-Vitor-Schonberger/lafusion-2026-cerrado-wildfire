"""
LAFusion 2026: Spatio-Temporal Information Fusion Framework
Module: src/evaluation/generate_maps.py
Description: Generates spatial cartographic risk maps and confusion matrix comparisons across the Brazilian Cerrado.
"""

import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import (
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    FIGURE_DPI,
)

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = FIGURE_DPI


def generate_spatial_maps_and_matrices():
    print("=" * 80)
    print("🗺️ [FASE 4.3] GERANDO MAPAS ESPACIAIS DE RISCO E MATRIZES DE CONFUSÃO (300 DPI)")
    print("=" * 80)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    preds_file = DATA_PROCESSED_DIR / "predictions_test_2024_2025.csv"
    print(f"1. Carregando dados de predições: {preds_file}")
    df = pd.read_csv(preds_file)

    # Filter for peak drought period (August/September 2024) for spatial case study
    case_study = df[(df["ano"] == 2024) & (df["mes"].isin([8, 9]))].copy()

    # Aggregate average risk per spatial grid cell during peak dry season
    spatial_agg = case_study.groupby(["grid_lat", "grid_lon"]).agg({
        "ground_truth": "mean",
        "Baseline_WeatherOnly": "mean",
        "LateFusion_DempsterShafer": "mean",
        "EarlyFusion_XGBoost": "mean",
    }).reset_index()

    # --------------------------------------------------------------------------
    # FIGURA 5: SPATIO-TEMPORAL RISK MAP COMPARISON (GOIÁS / DF)
    # --------------------------------------------------------------------------
    print("2. Gerando Figura 5: Mapa Espacial Comparativo de Risco no Cerrado (300 DPI)...")
    fig5_path_png = FIGURES_DIR / "fig5_spatiotemporal_wildfire_risk_map_goias.png"
    fig5_path_pdf = FIGURES_DIR / "fig5_spatiotemporal_wildfire_risk_map_goias.pdf"

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharex=True, sharey=True)

    panels = [
        ("ground_truth", "A) Ground Truth Active Fires", "Reds"),
        ("Baseline_WeatherOnly", "B) Weather-Only Baseline (INMET)", "YlOrRd"),
        ("LateFusion_DempsterShafer", "C) Late Fusion (Dempster-Shafer)", "YlOrRd"),
        ("EarlyFusion_XGBoost", "D) Early Feature Fusion (XGBoost)", "YlOrRd"),
    ]

    for ax, (col_name, title, cmap) in zip(axes, panels):
        scatter = ax.scatter(
            spatial_agg["grid_lon"],
            spatial_agg["grid_lat"],
            c=spatial_agg[col_name],
            cmap=cmap,
            s=85,
            edgecolor="black",
            linewidth=0.4,
            vmin=0.0,
            vmax=1.0,
            alpha=0.9
        )
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
        ax.set_xlabel("Longitude (°W)", fontsize=9.5)
        ax.grid(True, linestyle=":", alpha=0.6)

        # Highlight capital region (Goiânia & Brasília)
        ax.plot(-49.26, -16.68, "k*", markersize=8)  # Goiânia
        ax.plot(-47.89, -15.79, "k^", markersize=7)  # Brasília

    axes[0].set_ylabel("Latitude (°S)", fontsize=9.5)
    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), orientation="vertical", shrink=0.8, pad=0.02)
    cbar.set_label("Mean Predicted Risk / Fire Frequency (Aug-Sep 2024)", fontsize=10, fontweight="bold")

    plt.suptitle("Spatio-Temporal Wildfire Risk Forecast in the State of Goiás & DF (Peak Drought Case Study)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    plt.savefig(fig5_path_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.savefig(fig5_path_pdf, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Salva: {fig5_path_png}")

    # --------------------------------------------------------------------------
    # FIGURA 6: CONFUSION MATRICES COMPARISON
    # --------------------------------------------------------------------------
    print("3. Gerando Figura 6: Matrizes de Confusão Comparativas (300 DPI)...")
    fig6_path_png = FIGURES_DIR / "fig6_confusion_matrices_comparison.png"
    fig6_path_pdf = FIGURES_DIR / "fig6_confusion_matrices_comparison.pdf"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    models_cm = [
        ("Baseline_WeatherOnly", "A) Baseline: Weather-Only", 0.40),
        ("LateFusion_DempsterShafer", "B) Late Fusion: Dempster-Shafer", 0.40),
        ("EarlyFusion_XGBoost", "C) Early Fusion: XGBoost", 0.40),
    ]

    y_true = df["ground_truth"].values

    for ax, (m_col, title, thresh) in zip(axes, models_cm):
        y_pred = (df[m_col].values >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        # Normalize by row (True Class)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["No Risk", "Severe Risk"],
            yticklabels=["No Risk", "Severe Risk"],
            annot_kws={"fontsize": 11, "fontweight": "bold"}
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("True Class", fontsize=10)

    plt.suptitle("Normalized Confusion Matrices on the 2024-2025 Test Set (44,591 Samples)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()

    plt.savefig(fig6_path_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.savefig(fig6_path_pdf, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Salva: {fig6_path_png}")
    print(">>> [FASE 4.3] Mapas Espaciais e Matrizes de Confusão Concluídos!\n")
    return True


if __name__ == "__main__":
    generate_spatial_maps_and_matrices()
