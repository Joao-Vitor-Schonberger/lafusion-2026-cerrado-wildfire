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
    TRAIN_YEARS,
    TEST_YEARS,
    RANDOM_SEED,
)

from src.fusion.baselines import UnimodalBaselines
from src.fusion.early_fusion import EarlyFeatureFusion
from src.fusion.late_fusion import LateDecisionFusion


def run_training_pipeline():
    print("=" * 80)
    print("🚀 [FASE 3] INICIANDO TREINAMENTO DO ECOSSISTEMA DE MODELOS DE FUSÃO DE INFORMAÇÃO")
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

    print(f" - Conjunto de Treino ({TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}): {len(df_train):,} amostras (Positivos: {y_train.sum():,})")
    print(f" - Conjunto de Teste ({TEST_YEARS[0]}-{TEST_YEARS[-1]}):  {len(df_test):,} amostras (Positivos: {y_test.sum():,})")

    # Dictionary to collect all model predictions
    predictions_dict = {
        "cell_id": df_test["cell_id"].values,
        "data": df_test["data"].values,
        "ano": df_test["ano"].values,
        "mes": df_test["mes"].values,
        "grid_lat": df_test["grid_lat"].values,
        "grid_lon": df_test["grid_lon"].values,
        "ground_truth": y_test,
    }

    # --------------------------------------------------------------------------
    # 1. TREINANDO BASELINES UNIMODAIS (FONTE ÚNICA)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(">>> 1. Treinando Baselines de Fonte Única (Clima, Fogo e Solo)...")
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
    # 3. TREINANDO LATE DECISION-LEVEL FUSION (Dempster-Shafer, Soft-Voting, Stacking)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(">>> 3. Treinando Late Decision-Level Fusion (Teoria de Dempster-Shafer & Ensembles)...")
    late_fusion = LateDecisionFusion(random_state=RANDOM_SEED)
    late_fusion.fit(df_train, target_col=TARGET_COLUMN)
    preds_late = late_fusion.predict_proba(df_test)
    for model_name, proba in preds_late.items():
        predictions_dict[model_name] = proba

    # --------------------------------------------------------------------------
    # 4. AVALIAÇÃO DE DESEMPENHO E TABELA COMPARATIVA
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 RESULTADOS COMPARATIVOS NO CONJUNTO DE TESTE OUT-OF-TIME (2024-2025):")
    print("=" * 80)

    model_keys = list(preds_baselines.keys()) + list(preds_early.keys()) + list(preds_late.keys())
    results_rows = []

    for m_name in model_keys:
        proba = predictions_dict[m_name]
        # Optimal threshold = 0.40 to balance precision and recall on imbalanced wildfire data
        pred_binary = (proba >= 0.40).astype(int)

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
    # Sort by F1-Score descending
    df_results = df_results.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)

    # Print formatted table
    print(df_results.to_string(index=False))

    # --------------------------------------------------------------------------
    # 5. EXPORTAÇÃO DOS RESULTADOS (CSV, Parquet, LaTeX Table)
    # --------------------------------------------------------------------------
    # Save predictions
    df_preds = pd.DataFrame(predictions_dict)
    output_preds_csv = DATA_PROCESSED_DIR / "predictions_test_2024_2025.csv"
    output_preds_parquet = DATA_PROCESSED_DIR / "predictions_test_2024_2025.parquet"
    df_preds.to_csv(output_preds_csv, index=False)
    try:
        df_preds.to_parquet(output_preds_parquet, index=False)
    except Exception:
        pass

    # Save metrics table
    output_table_csv = TABLES_DIR / "table1_model_comparison.csv"
    output_table_tex = TABLES_DIR / "table1_model_comparison.tex"
    df_results.to_csv(output_table_csv, index=False)

    # Generate professional LaTeX table for Springer CCIS paper
    latex_table_str = df_results.to_latex(
        index=False,
        caption="Performance comparison between Unimodal Baselines, Early Feature-Level Fusion, and Late Decision-Level Fusion on the 2024-2025 Out-of-Time Test Set in the Brazilian Cerrado.",
        label="tab:model_comparison",
        position="htbp"
    )
    with open(output_table_tex, "w", encoding="utf-8") as f:
        f.write(latex_table_str)

    print("\n" + "=" * 80)
    print(f"✅ Predições salvas em: {output_preds_csv}")
    print(f"✅ Tabela comparativa (CSV) salva em: {output_table_csv}")
    print(f"✅ Tabela comparativa (LaTeX) salva em: {output_table_tex}")
    print("🏆 [FASE 3] Concluída com Sucesso!")
    print("=" * 80)
    return df_results


if __name__ == "__main__":
    run_training_pipeline()
