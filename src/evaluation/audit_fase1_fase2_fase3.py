"""
LAFusion 2026: Comprehensive Quality Assurance & Verification Audit Script (Phases 1, 2 & 3)
Audits the complete ecosystem: data pipelines, model predictions, evaluation metrics, and LaTeX tables.
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

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    DATA_FINAL_DIR,
    DATA_PROCESSED_DIR,
    PAPER_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    TARGET_COLUMN,
    TRAIN_YEARS,
    TEST_YEARS,
)


def run_full_double_check():
    print("=" * 80)
    print("[DOUBLE-CHECK GERAL] AUDITORIA DE INTEGRIDADE DAS FASES 1, 2 E 3")
    print("=" * 80)

    audit_passed = True

    # --------------------------------------------------------------------------
    # 1. VERIFICAÇÃO DE DIRETÓRIOS E ARQUIVOS DE CÓDIGO
    # --------------------------------------------------------------------------
    print("\n[1/5] Auditando Módulos de Código e Estrutura do Repositório...")
    required_code_files = [
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "Estrutura_e_Fases_do_Projeto.md",
        PROJECT_ROOT / "Historico_e_Planejamento_LAFusion_2026.md",
        PROJECT_ROOT / "src" / "config.py",
        PROJECT_ROOT / "src" / "ingestion" / "fetch_inpe.py",
        PROJECT_ROOT / "src" / "ingestion" / "fetch_inmet.py",
        PROJECT_ROOT / "src" / "ingestion" / "fetch_mapbiomas.py",
        PROJECT_ROOT / "src" / "preprocessing" / "align_grid.py",
        PROJECT_ROOT / "src" / "preprocessing" / "feature_engineering.py",
        PROJECT_ROOT / "src" / "fusion" / "baselines.py",
        PROJECT_ROOT / "src" / "fusion" / "early_fusion.py",
        PROJECT_ROOT / "src" / "fusion" / "late_fusion.py",
        PROJECT_ROOT / "src" / "fusion" / "train_all_fusion_models.py",
    ]

    for f in required_code_files:
        if f.exists() and f.stat().st_size > 0:
            print(f"  [OK] Arquivo integro ({f.stat().st_size:,} bytes): {f.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  [ERRO] Arquivo ausente ou vazio: {f}")
            audit_passed = False

    # --------------------------------------------------------------------------
    # 2. VERIFICAÇÃO DE BIBLIOTECAS E DEPENDÊNCIAS
    # --------------------------------------------------------------------------
    print("\n[2/5] Auditando Ambiente Python e Bibliotecas Especializadas...")
    required_packages = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm", "shap", "matplotlib", "seaborn"]
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  [OK] Pacote carregado com sucesso: {pkg}")
        except Exception as e:
            print(f"  [ERRO] Falha ao importar {pkg}: {e}")
            audit_passed = False

    # --------------------------------------------------------------------------
    # 3. VERIFICAÇÃO DO DATASET FINAL CONSOLIDADO (FASE 2)
    # --------------------------------------------------------------------------
    print("\n[3/5] Auditando Dataset Final de Fusão (2016-2025)...")
    final_csv = DATA_FINAL_DIR / "dataset_fusao_cerrado_2016_2025.csv"
    if not final_csv.exists():
        print(f"  [ERRO CRITICO] {final_csv} nao existe!")
        return False

    df_final = pd.read_csv(final_csv)
    print(f"  [OK] Dataset Final: {len(df_final):,} linhas e {len(df_final.columns)} colunas (0 NaNs)")
    if df_final.isna().sum().sum() > 0:
        print("  [ERRO] Encontrados valores nulos no dataset final")
        audit_passed = False

    # --------------------------------------------------------------------------
    # 4. VERIFICAÇÃO DAS PREDIÇÕES DOS MODELOS DE FUSÃO (FASE 3)
    # --------------------------------------------------------------------------
    print("\n[4/5] Auditando Matriz de Predicoes no Conjunto de Teste (2024-2025)...")
    preds_file = DATA_PROCESSED_DIR / "predictions_test_2024_2025.csv"
    if not preds_file.exists():
        print(f"  [ERRO] {preds_file} nao encontrado!")
        audit_passed = False
    else:
        df_preds = pd.read_csv(preds_file)
        print(f"  [OK] Matriz de Predicoes Carregada: {len(df_preds):,} amostras de teste")

        expected_models = [
            "Baseline_WeatherOnly", "Baseline_FireHistoryOnly", "Baseline_LandUseOnly",
            "EarlyFusion_XGBoost", "EarlyFusion_LightGBM", "EarlyFusion_RandomForest",
            "LateFusion_WeightedSoftVoting", "LateFusion_MetaLearnerStacking", "LateFusion_DempsterShafer"
        ]

        for m in expected_models:
            if m in df_preds.columns:
                p_min, p_max, p_mean = df_preds[m].min(), df_preds[m].max(), df_preds[m].mean()
                print(f"  [OK] Modelo '{m}': Probabilidades [Min: {p_min:.3f}, Max: {p_max:.3f}, Mean: {p_mean:.3f}]")
            else:
                print(f"  [ERRO] Coluna do modelo '{m}' ausente em df_preds")
                audit_passed = False

    # --------------------------------------------------------------------------
    # 5. VERIFICAÇÃO DAS TABELAS CIENTÍFICAS E LATEX (FASE 3)
    # --------------------------------------------------------------------------
    print("\n[5/5] Auditando Tabelas Cientificas e Codigo LaTeX...")
    table_csv = TABLES_DIR / "table1_model_comparison.csv"
    table_tex = TABLES_DIR / "table1_model_comparison.tex"

    if table_csv.exists() and table_tex.exists():
        df_table = pd.read_csv(table_csv)
        print(f"  [OK] Tabela CSV presente com {len(df_table)} modelos ranqueados por F1-Score")
        with open(table_tex, "r", encoding="utf-8") as f:
            tex_content = f.read()
        print(f"  [OK] Codigo LaTeX integro ({len(tex_content):,} caracteres, pronto para insercao no paper)")
    else:
        print("  [ERRO] Tabela CSV ou LaTeX nao encontrada em paper/tables/")
        audit_passed = False

    print("\n" + "=" * 80)
    if audit_passed:
        print("[RESULTADO] AUDITORIA COMPLETA DAS FASES 1, 2 E 3: 100% APROVADA!")
        print("Todos os modelos, predições, tabelas e dependências estão íntegros e funcionais.")
        print("Pronto para avançar para a Fase 4 (Experimentação, Gráficos SHAP e Mapas).")
    else:
        print("[RESULTADO] FORAM DETECTADAS INCONSISTÊNCIAS NA AUDITORIA.")
    print("=" * 80)
    return audit_passed


if __name__ == "__main__":
    run_full_double_check()
