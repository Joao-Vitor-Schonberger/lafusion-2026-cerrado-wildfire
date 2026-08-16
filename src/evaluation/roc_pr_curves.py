"""
LAFusion 2026: Spatio-Temporal Information Fusion Framework
Module: src/evaluation/roc_pr_curves.py
Description: Generates high-resolution publication figures (300 DPI) for ROC and Precision-Recall curves.
Compares Unimodal Baselines vs Early Feature-Level Fusion vs Late Decision-Level Fusion (Dempster-Shafer).
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
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

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


def plot_curves():
    print("=" * 80)
    print("📈 [FASE 4.2] GERANDO CURVAS ROC E PRECISION-RECALL (300 DPI)")
    print("=" * 80)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    preds_file = DATA_PROCESSED_DIR / "predictions_test_2024_2025.csv"
    print(f"1. Carregando predições de teste: {preds_file}")
    df = pd.read_csv(preds_file)

    y_true = df["ground_truth"].values

    # Key models to highlight
    models_to_plot = [
        ("EarlyFusion_RandomForest", "Early Fusion: Random Forest", "#1f77b4", "-", 2.2),
        ("EarlyFusion_XGBoost", "Early Fusion: XGBoost", "#2ca02c", "-", 2.0),
        ("LateFusion_MetaLearnerStacking", "Late Fusion: Meta-Learner Stacking", "#9467bd", "-", 1.8),
        ("LateFusion_DempsterShafer", "Late Fusion: Dempster-Shafer (DST)", "#d62728", "--", 2.0),
        ("LateFusion_WeightedSoftVoting", "Late Fusion: Weighted Soft-Voting", "#ff7f0e", "--", 1.6),
        ("Baseline_WeatherOnly", "Baseline: Weather-Only (INMET)", "#8c564b", ":", 1.8),
        ("Baseline_LandUseOnly", "Baseline: LandUse-Only (MapBiomas)", "#7f7f7f", ":", 1.5),
        ("Baseline_FireHistoryOnly", "Baseline: Fire-History-Only (INPE)", "#bcbd22", ":", 1.5),
    ]

    # --------------------------------------------------------------------------
    # FIGURA 3: ROC CURVES COMPARISON
    # --------------------------------------------------------------------------
    print("2. Gerando Figura 3: Curvas ROC Comparativas (300 DPI)...")
    fig3_path_png = FIGURES_DIR / "fig3_roc_curves_comparison.png"
    fig3_path_pdf = FIGURES_DIR / "fig3_roc_curves_comparison.pdf"

    plt.figure(figsize=(8.5, 6.5))

    for col_name, label_name, color, linestyle, lw in models_to_plot:
        if col_name in df.columns:
            fpr, tpr, _ = roc_curve(y_true, df[col_name].values)
            roc_auc_val = auc(fpr, tpr)
            plt.plot(
                fpr, tpr,
                color=color, linestyle=linestyle, linewidth=lw,
                label=f"{label_name} (AUC = {roc_auc_val:.4f})"
            )

    plt.plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.4, label="Random Chance (AUC = 0.5000)")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.02])
    plt.xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (TPR / Recall)", fontsize=11, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) on 2024-2025 Test Set", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(fig3_path_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.savefig(fig3_path_pdf, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Salva: {fig3_path_png}")

    # --------------------------------------------------------------------------
    # FIGURA 4: PRECISION-RECALL CURVES
    # --------------------------------------------------------------------------
    print("3. Gerando Figura 4: Curvas Precision-Recall (300 DPI)...")
    fig4_path_png = FIGURES_DIR / "fig4_precision_recall_curves.png"
    fig4_path_pdf = FIGURES_DIR / "fig4_precision_recall_curves.pdf"

    plt.figure(figsize=(8.5, 6.5))

    for col_name, label_name, color, linestyle, lw in models_to_plot:
        if col_name in df.columns:
            prec, rec, _ = precision_recall_curve(y_true, df[col_name].values)
            ap_val = average_precision_score(y_true, df[col_name].values)
            plt.plot(
                rec, prec,
                color=color, linestyle=linestyle, linewidth=lw,
                label=f"{label_name} (PR-AUC = {ap_val:.4f})"
            )

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.02])
    plt.xlabel("Recall", fontsize=11, fontweight="bold")
    plt.ylabel("Precision", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall Curves on Imbalanced 2024-2025 Test Set", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(fig4_path_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.savefig(fig4_path_pdf, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Salva: {fig4_path_png}")
    print(">>> [FASE 4.2] Curvas ROC & PR Concluídas!\n")
    return True


if __name__ == "__main__":
    plot_curves()
