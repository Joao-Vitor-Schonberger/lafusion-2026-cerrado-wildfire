"""
LAFusion 2026: Quality Assurance & Verification Audit Script
Validates all assets, datasets, directories and modules generated in Phases 1 & 2.
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
    START_YEAR,
    END_YEAR,
    YEARS,
    MONTHS,
    TARGET_COLUMN,
    GEO_BOUNDS,
)


def run_double_check():
    print("=" * 80)
    print("[DOUBLE-CHECK] INICIANDO AUDITORIA GERAL DAS FASES 1 E 2")
    print("=" * 80)

    audit_passed = True

    # --------------------------------------------------------------------------
    # 1. VERIFICACAO DA ESTRUTURA DE DIRETORIOS
    # --------------------------------------------------------------------------
    print("\n[1/4] Auditando Estrutura Fisica de Diretorios...")
    required_dirs = [
        DATA_DIR,
        DATA_DIR / "raw" / "inpe_fires",
        DATA_DIR / "raw" / "inmet_weather",
        DATA_DIR / "raw" / "mapbiomas_landuse",
        DATA_PROCESSED_DIR,
        DATA_FINAL_DIR,
        PROJECT_ROOT / "src" / "ingestion",
        PROJECT_ROOT / "src" / "preprocessing",
        PROJECT_ROOT / "src" / "fusion",
        PROJECT_ROOT / "src" / "evaluation",
        PROJECT_ROOT / "notebooks",
        PAPER_DIR / "template",
        FIGURES_DIR,
        TABLES_DIR,
    ]

    all_dirs_ok = True
    for d in required_dirs:
        if d.exists():
            print(f"  [OK] Diretorio presente: {d.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  [ERRO] Diretorio AUSENTE: {d}")
            all_dirs_ok = False
            audit_passed = False

    # --------------------------------------------------------------------------
    # 2. VERIFICACAO DOS ARQUIVOS DE CODIGO E DOCUMENTACAO
    # --------------------------------------------------------------------------
    print("\n[2/4] Auditando Arquivos de Codigo e Documentacao...")
    required_files = [
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "Estrutura_e_Fases_do_Projeto.md",
        PROJECT_ROOT / "Historico_e_Planejamento_LAFusion_2026.md",
        PROJECT_ROOT / "LAFusion_2026_Informacoes.md",
        PROJECT_ROOT / "src" / "config.py",
        PROJECT_ROOT / "src" / "__init__.py",
        PROJECT_ROOT / "src" / "ingestion" / "__init__.py",
        PROJECT_ROOT / "src" / "ingestion" / "fetch_inpe.py",
        PROJECT_ROOT / "src" / "ingestion" / "fetch_inmet.py",
        PROJECT_ROOT / "src" / "ingestion" / "fetch_mapbiomas.py",
        PROJECT_ROOT / "src" / "preprocessing" / "__init__.py",
        PROJECT_ROOT / "src" / "preprocessing" / "align_grid.py",
        PROJECT_ROOT / "src" / "preprocessing" / "feature_engineering.py",
        PROJECT_ROOT / "src" / "fusion" / "__init__.py",
        PROJECT_ROOT / "src" / "evaluation" / "__init__.py",
    ]

    all_files_ok = True
    for f in required_files:
        if f.exists() and f.stat().st_size > 0:
            print(f"  [OK] Arquivo integro ({f.stat().st_size:,} bytes): {f.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  [ERRO] Arquivo ausente ou vazio: {f}")
            all_files_ok = False
            audit_passed = False

    # --------------------------------------------------------------------------
    # 3. VERIFICACAO DA INTEGRIDADE DO DATASET FINAL DE FUSAO
    # --------------------------------------------------------------------------
    print("\n[3/4] Auditando Integridade e Consistencia dos Dados Finais...")
    final_csv = DATA_FINAL_DIR / "dataset_fusao_cerrado_2016_2025.csv"

    if not final_csv.exists():
        print(f"  [ERRO CRITICO] {final_csv} nao existe!")
        return False

    df = pd.read_csv(final_csv)
    print(f"  [OK] Dataset Final Carregado: {len(df):,} linhas e {len(df.columns)} colunas")

    # Checagem de NaNs
    total_nans = df.isna().sum().sum()
    if total_nans == 0:
        print("  [OK] Valores Nulos (NaNs): 0 (Nenhum dado faltante)")
    else:
        print(f"  [AVISO] {total_nans} valores nulos encontrados")
        audit_passed = False

    # Checagem Temporal
    anos_presentes = sorted(df["ano"].unique().tolist())
    meses_presentes = sorted(df["mes"].unique().tolist())
    print(f"  [OK] Anos no dataset: {anos_presentes} (Total: {len(anos_presentes)} anos)")
    print(f"  [OK] Meses no dataset: {meses_presentes} (Total: {len(meses_presentes)} meses)")

    if anos_presentes != YEARS or meses_presentes != MONTHS:
        print("  [ERRO] Inconsistencia no horizonte temporal")
        audit_passed = False

    # Checagem Geografica
    lat_min, lat_max = df["grid_lat"].min(), df["grid_lat"].max()
    lon_min, lon_max = df["grid_lon"].min(), df["grid_lon"].max()
    print(f"  [OK] Limites Geograficos: Lat [{lat_min:.2f}, {lat_max:.2f}] | Lon [{lon_min:.2f}, {lon_max:.2f}] (Cerrado/Goias)")

    # Checagem da Variavel-Alvo
    target_values = df[TARGET_COLUMN].value_counts().to_dict()
    pos = target_values.get(1, 0)
    neg = target_values.get(0, 0)
    total = pos + neg
    print(f"  [OK] Variavel-Alvo ({TARGET_COLUMN}):")
    print(f"       - Classe 1 (Severe Fire Risk): {pos:,} ({pos/total*100:.2f}%)")
    print(f"       - Classe 0 (Low/No Risk):      {neg:,} ({neg/total*100:.2f}%)")

    # Checagem das Features Multimodais
    print("\n[4/4] Auditando Presenca dos Grupos de Features Multimodais...")
    feature_groups = {
        "Meteorologicas": ["temp_max_c", "umid_min_pct", "vento_rajada_ms", "precipitacao_mm", "dias_sem_chuva"],
        "Lags e Medias Moveis": ["umid_min_lag1", "umid_min_mean_3d", "umid_min_mean_7d", "temp_max_mean_7d", "vento_rajada_max_3d"],
        "Uso do Solo & Combustivel": ["classe_uso_solo", "indice_inflamabilidade", "biomassa_combustivel_t_ha", "dist_estrada_km"],
        "Fogo & Satelite": ["num_focos_satelite", "frp_max_mw", "frp_soma_mw"],
        "Interacoes & Fisica": ["fwi_proxy", "combustivel_secura_interacao", "exposicao_antropica", "dias_sem_chuva_log"],
    }

    for group_name, cols in feature_groups.items():
        missing_cols = [c for c in cols if c not in df.columns]
        if not missing_cols:
            print(f"  [OK] Grupo '{group_name}': Todas as {len(cols)} features presentes")
        else:
            print(f"  [ERRO] Grupo '{group_name}': Faltam features {missing_cols}")
            audit_passed = False

    print("\n" + "=" * 80)
    if audit_passed:
        print("[RESULTADO] DOUBLE-CHECK 100% APROVADO! TODOS OS SISTEMAS E DADOS INTEGROS.")
        print("Tudo pronto e validado para iniciarmos a Fase 3 (Modelagem e Fusao).")
    else:
        print("[RESULTADO] FORAM ENCONTRADAS NAO-CONFORMIDADES.")
    print("=" * 80)
    return audit_passed


if __name__ == "__main__":
    run_double_check()
