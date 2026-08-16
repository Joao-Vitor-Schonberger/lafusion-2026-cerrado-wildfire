"""
LAFusion 2026: Spatio-Temporal Information Fusion Framework
Module: src/evaluation/shap_analysis.py
Description: Explainable AI (XAI) using TreeSHAP to quantify the impact of each multimodal feature
on severe wildfire risk prediction in the Brazilian Cerrado.
"""

import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import (
    DATA_FINAL_DIR,
    FIGURES_DIR,
    TARGET_COLUMN,
    TRAIN_YEARS,
    TEST_YEARS,
    RANDOM_SEED,
    FIGURE_DPI,
)

# Publication style configuration
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = FIGURE_DPI


def run_shap_analysis():
    print("=" * 80)
    print("🧠 [FASE 4.1] EXPLICABILIDADE DE IA COM SHAP (SHAPLEY ADDITIVE EXPLANATIONS)")
    print("=" * 80)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    input_file = DATA_FINAL_DIR / "dataset_fusao_cerrado_2016_2025.csv"
    print(f"1. Carregando dados para análise SHAP: {input_file}")
    df = pd.read_csv(input_file)

    exclude_cols = ["cell_id", "data", "classe_uso_solo", "estacao_mais_proxima", TARGET_COLUMN]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    df_train = df[df["ano"].isin(TRAIN_YEARS)].copy().reset_index(drop=True)
    df_test = df[df["ano"].isin(TEST_YEARS)].copy().reset_index(drop=True)

    X_train = df_train[feature_cols].values
    y_train = df_train[TARGET_COLUMN].values

    # Sample test set for fast and precise TreeSHAP computation
    sample_size = min(3000, len(df_test))
    df_test_sample = df_test.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
    X_test_sample = df_test_sample[feature_cols].values

    print(f"2. Treinando XGBoost Early Fusion para extração dos SHAP Values...")
    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    print("3. Calculando TreeSHAP Values nas amostras de teste...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)

    # Feature name friendly mapping for publication
    friendly_feature_names = {
        "fwi_proxy": "Fire Weather Index (FWI Proxy)",
        "combustivel_secura_interacao": "Fuel-Dryness Interaction Index",
        "umid_min_pct": "Daily Min Relative Humidity (%)",
        "umid_min_mean_7d": "7-Day Mean Min Humidity (%)",
        "dias_sem_chuva": "Consecutive Dry Days (DSR)",
        "dias_sem_chuva_log": "Log(Consecutive Dry Days)",
        "vento_rajada_ms": "Max Wind Gust Speed (m/s)",
        "vento_rajada_max_3d": "3-Day Max Wind Gust (m/s)",
        "temp_max_c": "Daily Max Temperature (°C)",
        "temp_max_mean_7d": "7-Day Mean Max Temp (°C)",
        "indice_inflamabilidade": "Land Cover Flammability Index",
        "biomassa_combustivel_t_ha": "Fuel Biomass Density (t/ha)",
        "dist_estrada_km": "Distance to Road/Crop Edge (km)",
        "exposicao_antropica": "Anthropic Exposure Index",
        "num_focos_satelite": "Satellite Thermal Detections",
        "frp_max_mw": "Max Fire Radiative Power (MW)",
        "frp_soma_mw": "Sum Fire Radiative Power (MW)",
        "precipitacao_mm": "Daily Precipitation (mm)",
        "precipitacao_acum_7d": "7-Day Accumulated Rain (mm)",
        "classe_id": "Land Cover Class ID",
    }

    display_names = [friendly_feature_names.get(c, c) for c in feature_cols]
    df_display = pd.DataFrame(X_test_sample, columns=display_names)

    # --------------------------------------------------------------------------
    # FIGURA 1: SHAP SUMMARY BEESWARM PLOT
    # --------------------------------------------------------------------------
    print("4. Gerando Figura 1: SHAP Summary Beeswarm Plot (300 DPI)...")
    fig1_path_png = FIGURES_DIR / "fig1_shap_summary_beeswarm.png"
    fig1_path_pdf = FIGURES_DIR / "fig1_shap_summary_beeswarm.pdf"

    plt.figure(figsize=(10, 6.5))
    shap.summary_plot(
        shap_values,
        df_display,
        max_display=12,
        show=False,
        plot_type="dot"
    )
    plt.title("TreeSHAP Feature Impact on Severe Wildfire Risk (2024-2025 Test Set)", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("SHAP value (impact on model log-odds output)", fontsize=10)
    plt.tight_layout()
    plt.savefig(fig1_path_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.savefig(fig1_path_pdf, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Salva: {fig1_path_png}")

    # --------------------------------------------------------------------------
    # FIGURA 2: SHAP MEAN ABSOLUTE IMPORTANCE BAR PLOT
    # --------------------------------------------------------------------------
    print("5. Gerando Figura 2: SHAP Feature Importance Ranking (300 DPI)...")
    fig2_path_png = FIGURES_DIR / "fig2_shap_feature_importance_bar.png"
    fig2_path_pdf = FIGURES_DIR / "fig2_shap_feature_importance_bar.pdf"

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df_imp = pd.DataFrame({"feature": display_names, "importance": mean_abs_shap})
    df_imp = df_imp.sort_values(by="importance", ascending=True).tail(12)

    plt.figure(figsize=(9, 5.5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_imp)))
    bars = plt.barh(df_imp["feature"], df_imp["importance"], color=colors, edgecolor="black", linewidth=0.6)
    plt.xlabel("Mean |SHAP value| (Average Impact on Prediction)", fontsize=10, fontweight="bold")
    plt.title("Global Feature Importance Ranking via TreeSHAP in the Cerrado", fontsize=12, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.02, bar.get_y() + bar.get_height() / 2.0, f"{w:.2f}", ha="left", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(fig2_path_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.savefig(fig2_path_pdf, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Salva: {fig2_path_png}")
    print(">>> [FASE 4.1] SHAP Explainability Concluída!\n")
    return True


if __name__ == "__main__":
    run_shap_analysis()
