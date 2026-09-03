"""
LAFusion 2026: Spatio-Temporal Information Fusion Framework
Module: src/fusion/train_all_fusion_models.py
Description: Master orchestration pipeline for training and evaluating all 9 information fusion models.
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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import (
    DATA_FINAL_DIR,
    DATA_PROCESSED_DIR,
    TABLES_DIR,
    TARGET_COLUMN,
    TARGET_COLUMN_48H,
    TARGET_COLUMN_72H,
    TRAIN_YEARS,
    TEST_YEARS,
    RANDOM_SEED,
)

from src.fusion.baselines import UnimodalBaselines
from src.fusion.early_fusion import EarlyFeatureFusion
from src.fusion.late_fusion import LateDecisionFusion


def run_training_pipeline():
    print("=" * 80)
    print("🚀 [FASE 3] TREINAMENTO MULTI-MODELO E MULTI-HORIZONTE (SEM DATA LEAKAGE)")
    print("=" * 80)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    input_file = DATA_FINAL_DIR / "dataset_fusao_cerrado_2016_2025.csv"
    print(f"1. Carregando dataset consolidado: {input_file}")
    df = pd.read_csv(input_file)

    # Temporal split: Train on 2016-2023, Test on 2024-2025
    df_train = df[df["ano"].isin(TRAIN_YEARS)].copy().reset_index(drop=True)
    df_test = df[df["ano"].isin(TEST_YEARS)].copy().reset_index(drop=True)

    y_train = df_train[TARGET_COLUMN].values
    y_test = df_test[TARGET_COLUMN].values

    print(f" - Conjunto de Treino ({TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}): {len(df_train):,} amostras (Positivos 24h: {y_train.sum():,} | {y_train.sum()/len(df_train)*100:.2f}%)")
    print(f" - Conjunto de Teste ({TEST_YEARS[0]}-{TEST_YEARS[-1]}):  {len(df_test):,} amostras (Positivos 24h: {y_test.sum():,} | {y_test.sum()/len(df_test)*100:.2f}%)")

    # Dictionary to collect all model predictions on test set
    predictions_dict = {
        "cell_id": df_test["cell_id"].values,
        "data": df_test["data"].values,
        "ano": df_test["ano"].values,
        "mes": df_test["mes"].values,
        "grid_lat": df_test["grid_lat"].values,
        "grid_lon": df_test["grid_lon"].values,
        "ground_truth": y_test,
        "ground_truth_48h": df_test[TARGET_COLUMN_48H].values,
        "ground_truth_72h": df_test[TARGET_COLUMN_72H].values,
    }

    # --------------------------------------------------------------------------
    # 1. TREINANDO BASELINES UNIMODAIS (FONTE ÚNICA)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(">>> 1. Treinando Baselines de Fonte Única (Clima, Fogo Passado, Uso do Solo)...")
    baselines = UnimodalBaselines(random_state=RANDOM_SEED)
    baselines.fit(df_train, target_col=TARGET_COLUMN)
    preds_baselines = baselines.predict_proba(df_test)
    for model_name, proba in preds_baselines.items():
        predictions_dict[model_name] = proba

    # --------------------------------------------------------------------------
    # 2. TREINANDO EARLY FEATURE-LEVEL FUSION (XGBoost, LightGBM, Random Forest)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(">>> 2. Treinando Early Feature-Level Fusion (Multimodal Matrix)...")
    early_fusion = EarlyFeatureFusion(random_state=RANDOM_SEED)
    early_fusion.fit(df_train, target_col=TARGET_COLUMN)
    preds_early = early_fusion.predict_proba(df_test)
    for model_name, proba in preds_early.items():
        predictions_dict[model_name] = proba

    # --------------------------------------------------------------------------
    # 3. TREINANDO LATE DECISION-LEVEL FUSION (Dempster-Shafer Dinâmico & Ensembles)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(">>> 3. Treinando Late Decision-Level Fusion (DST com Confiabilidade Dinâmica)...")
    late_fusion = LateDecisionFusion(random_state=RANDOM_SEED)
    late_fusion.fit(df_train, target_col=TARGET_COLUMN)
    preds_late = late_fusion.predict_proba(df_test, horizon=1)
    for model_name, proba in preds_late.items():
        predictions_dict[model_name] = proba

    # --------------------------------------------------------------------------
    # 4. AVALIAÇÃO DE DESEMPENHO NO HORIZONTE PRINCIPAL (24h)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 RESULTADOS NO CONJUNTO DE TESTE OUT-OF-TIME (HORIZONTE 24h / 2024-2025):")
    print("=" * 80)

    model_keys = list(preds_baselines.keys()) + list(preds_early.keys()) + list(preds_late.keys())
    results_rows = []

    for m_name in model_keys:
        proba = predictions_dict[m_name]
        # Operational threshold = 0.35 to balance early detection on imbalanced wildfire data
        pred_binary = (proba >= 0.35).astype(int)

        acc = accuracy_score(y_test, pred_binary)
        prec = precision_score(y_test, pred_binary, zero_division=0)
        rec = recall_score(y_test, pred_binary, zero_division=0)
        f1 = f1_score(y_test, pred_binary, zero_division=0)
        auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)

        # Categorize paradigm
        if "Baseline" in m_name:
            paradigm = "Single-Source Baseline"
        elif "EarlyFusion" in m_name:
            paradigm = "Early Feature Fusion"
        else:
            paradigm = "Late Decision Fusion"

        results_rows.append({
            "Paradigm": paradigm,
            "Model Architecture": m_name.replace("Baseline_", "").replace("EarlyFusion_", "").replace("LateFusion_", ""),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "PR-AUC": round(pr_auc, 4),
        })

    df_results = pd.DataFrame(results_rows)
    df_results = df_results.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    print(df_results.to_string(index=False))

    # --------------------------------------------------------------------------
    # 5. AVALIAÇÃO MULTI-HORIZONTE (DEGRADAÇÃO 24h -> 48h -> 72h)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("⏱️ AVALIAÇÃO MULTI-HORIZONTE (DEGRADAÇÃO PREDITIVA 24h, 48h, 72h):")
    print("=" * 80)

    horizons_eval = [
        (1, TARGET_COLUMN, "24h (t+1)"),
        (2, TARGET_COLUMN_48H, "48h (t+2)"),
        (3, TARGET_COLUMN_72H, "72h (t+3)"),
    ]

    key_models = [
        ("Early Fusion: XGBoost", EarlyFeatureFusion(random_state=RANDOM_SEED)),
        ("Late Fusion: Dempster-Shafer", LateDecisionFusion(random_state=RANDOM_SEED)),
    ]

    horizon_rows = []
    for h_num, h_col, h_label in horizons_eval:
        y_tr_h = df_train[h_col].values
        y_te_h = df_test[h_col].values

        # XGBoost Early Fusion
        ef_h = EarlyFeatureFusion(random_state=RANDOM_SEED)
        ef_h.fit(df_train, target_col=h_col)
        p_xgb_h = ef_h.predict_proba(df_test)["EarlyFusion_XGBoost"]
        bin_xgb_h = (p_xgb_h >= 0.35).astype(int)

        horizon_rows.append({
            "Horizon": h_label,
            "Model": "Early Fusion (XGBoost)",
            "Precision": round(precision_score(y_te_h, bin_xgb_h, zero_division=0), 4),
            "Recall": round(recall_score(y_te_h, bin_xgb_h, zero_division=0), 4),
            "F1-Score": round(f1_score(y_te_h, bin_xgb_h, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_te_h, p_xgb_h), 4),
        })

        # Dempster-Shafer Late Fusion
        lf_h = LateDecisionFusion(random_state=RANDOM_SEED)
        lf_h.fit(df_train, target_col=h_col)
        p_dst_h = lf_h.predict_proba(df_test, horizon=h_num)["LateFusion_DempsterShafer"]
        bin_dst_h = (p_dst_h >= 0.35).astype(int)

        horizon_rows.append({
            "Horizon": h_label,
            "Model": "Late Fusion (Dempster-Shafer)",
            "Precision": round(precision_score(y_te_h, bin_dst_h, zero_division=0), 4),
            "Recall": round(recall_score(y_te_h, bin_dst_h, zero_division=0), 4),
            "F1-Score": round(f1_score(y_te_h, bin_dst_h, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_te_h, p_dst_h), 4),
        })

    df_horizon = pd.DataFrame(horizon_rows)
    print(df_horizon.to_string(index=False))

    # --------------------------------------------------------------------------
    # 6. EXPORTAÇÃO DOS ARTEFATOS
    # --------------------------------------------------------------------------
    df_preds = pd.DataFrame(predictions_dict)
    output_preds_csv = DATA_PROCESSED_DIR / "predictions_test_2024_2025.csv"
    output_preds_parquet = DATA_PROCESSED_DIR / "predictions_test_2024_2025.parquet"
    df_preds.to_csv(output_preds_csv, index=False)
    try:
        df_preds.to_parquet(output_preds_parquet, index=False)
    except Exception:
        pass

    # Save Table 1: Model Comparison (24h)
    output_table_csv = TABLES_DIR / "table1_model_comparison.csv"
    output_table_tex = TABLES_DIR / "table1_model_comparison.tex"
    df_results.to_csv(output_table_csv, index=False)

    latex_table_str = df_results.to_latex(
        index=False,
        caption="Comparative performance across Unimodal Baselines, Early Feature Fusion, and Late Evidential Decision Fusion on the 2024--2025 Out-of-Time Test Set (24h Lead Time).",
        label="tab:model_comparison",
        position="htbp"
    )
    with open(output_table_tex, "w", encoding="utf-8") as f:
        f.write(latex_table_str)

    # Save Table 2: Multi-Horizon Degradation
    output_h_csv = TABLES_DIR / "table2_multi_horizon_comparison.csv"
    output_h_tex = TABLES_DIR / "table2_multi_horizon_comparison.tex"
    df_horizon.to_csv(output_h_csv, index=False)
    latex_h_str = df_horizon.to_latex(
        index=False,
        caption="Multi-Horizon Forecast Degradation (24h, 48h, and 72h Lead Times) for Early Fusion and Dempster-Shafer Late Fusion.",
        label="tab:multi_horizon",
        position="htbp"
    )
    with open(output_h_tex, "w", encoding="utf-8") as f:
        f.write(latex_h_str)

    print("\n" + "=" * 80)
    print(f"✅ Predições salvas em: {output_preds_csv}")
    print(f"✅ Tabela comparativa 24h salva em: {output_table_csv} e {output_table_tex}")
    print(f"✅ Tabela multi-horizonte salva em: {output_h_csv} e {output_h_tex}")
    print("🏆 [FASE 3] Concluída com Sucesso!")
    print("=" * 80)
    return df_results, df_horizon


if __name__ == "__main__":
    run_training_pipeline()

