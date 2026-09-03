"""
LAFusion 2026: Spatio-Temporal Information Fusion Framework
Module: src/preprocessing/feature_engineering.py
Description: Feature Engineering and Multi-Horizon Target Formulation without Data Leakage.
Produces rigorous out-of-time datasets with 24h, 48h and 72h proactive forecasting targets.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import (
    DATA_PROCESSED_DIR,
    DATA_FINAL_DIR,
    TARGET_COLUMN,
    TARGET_COLUMN_48H,
    TARGET_COLUMN_72H,
    CRITICAL_FRP_THRESHOLD,
    RANDOM_SEED,
)

np.random.seed(RANDOM_SEED + 40)


def build_feature_engineering_pipeline():
    """
    Applies rigorous feature engineering across temporal, meteorological and fuel dimensions.
    Constructs multi-horizon forecasting targets (t+1, t+2, t+3) and strictly separates
    past predictor features (up to t) from future wildfire occurrences (t+k).
    """
    print("=" * 75)
    print(">>> [FASE 2.5] FEATURE ENGINEERING E CONSTRUÇÃO DE ALVOS MULTI-HORIZONTE (SEM LEAKAGE)")
    print("=" * 75)

    DATA_FINAL_DIR.mkdir(parents=True, exist_ok=True)

    input_file = DATA_PROCESSED_DIR / "grid_aligned_spatiotemporal.csv"
    output_final_csv = DATA_FINAL_DIR / "dataset_fusao_cerrado_2016_2025.csv"
    output_final_parquet = DATA_FINAL_DIR / "dataset_fusao_cerrado_2016_2025.parquet"

    print(f"1. Carregando dados da grade espaço-temporal: {input_file}")
    df = pd.read_csv(input_file)
    print(f" - Registros originais: {len(df):,}")

    df["data"] = pd.to_datetime(df["data"])
    df = df.sort_values(["cell_id", "data"]).reset_index(drop=True)
    grouped = df.groupby("cell_id")

    print("2. Construindo Variáveis Preditivas em tempo t (Sem variáveis do futuro)...")
    
    # --------------------------------------------------------------------------
    # A) Meteorologia de Superfície & Lags Históricos (INMET)
    # --------------------------------------------------------------------------
    df["umid_min_lag1"] = grouped["umid_min_pct"].shift(1).bfill()
    df["umid_min_lag2"] = grouped["umid_min_pct"].shift(2).bfill()
    df["umid_min_lag3"] = grouped["umid_min_pct"].shift(3).bfill()

    df["temp_max_lag1"] = grouped["temp_max_c"].shift(1).bfill()
    df["temp_max_lag2"] = grouped["temp_max_c"].shift(2).bfill()
    df["vento_rajada_lag1"] = grouped["vento_rajada_ms"].shift(1).bfill()

    # Janelas móveis de estresse hídrico acumulado
    df["umid_min_mean_3d"] = grouped["umid_min_pct"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["umid_min_mean_7d"] = grouped["umid_min_pct"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["temp_max_mean_3d"] = grouped["temp_max_c"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["temp_max_mean_7d"] = grouped["temp_max_c"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["vento_rajada_max_3d"] = grouped["vento_rajada_ms"].transform(lambda x: x.rolling(3, min_periods=1).max())
    df["precipitacao_acum_7d"] = grouped["precipitacao_mm"].transform(lambda x: x.rolling(7, min_periods=1).sum())

    # Índices físicos em tempo t
    df["fwi_proxy"] = (df["temp_max_c"] * df["vento_rajada_ms"]) / (df["umid_min_pct"] + 1.0)
    df["combustivel_secura_interacao"] = df["indice_inflamabilidade"] * ((100.0 - df["umid_min_pct"]) / 100.0)
    df["exposicao_antropica"] = df["biomassa_combustivel_t_ha"] / (df["dist_estrada_km"] + 1.0)
    df["dias_sem_chuva_log"] = np.log1p(df["dias_sem_chuva"])

    # Componentes periódicos sazonais
    df["sin_dia_ano"] = np.sin(2 * np.pi * df["dia_do_ano"] / 365.25)
    df["cos_dia_ano"] = np.cos(2 * np.pi * df["dia_do_ano"] / 365.25)

    # --------------------------------------------------------------------------
    # B) Memória Histórica de Focos de Satélite (Exclusivamente Passada: t-1, t-2)
    # --------------------------------------------------------------------------
    # O modelo NÃO vê satélite contemporâneo do dia t para prever t+1, t+2 ou t+3.
    # Ele usa apenas histórico prévio para evitar dependência de detecção simultânea.
    df["focos_satelite_lag1"] = grouped["num_focos_satelite"].shift(1).fillna(0)
    df["focos_satelite_lag2"] = grouped["num_focos_satelite"].shift(2).fillna(0)
    df["focos_acum_3d"] = grouped["num_focos_satelite"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum()).fillna(0)
    df["frp_max_lag1"] = grouped["frp_max_mw"].shift(1).fillna(0)
    df["frp_soma_lag1"] = grouped["frp_soma_mw"].shift(1).fillna(0)

    print("3. Definindo Alvos Futuros Não-Circulares (t+1: 24h | t+2: 48h | t+3: 72h)...")
    # Ocorrência real de incêndio severo no dia avaliado:
    # Registra presença de múltiplos focos ou foco com FRP elevado (alta energia liberada)
    severe_event_raw = (
        (df["num_focos_satelite"] >= 2) |
        ((df["num_focos_satelite"] >= 1) & (df["frp_max_mw"] >= CRITICAL_FRP_THRESHOLD))
    ).astype(int)

    df["severe_raw"] = severe_event_raw

    # Criando os alvos com avanço temporal estrito (shift negativo por célula)
    df[TARGET_COLUMN] = grouped["severe_raw"].shift(-1)      # 24h ahead (t+1)
    df[TARGET_COLUMN_48H] = grouped["severe_raw"].shift(-2)  # 48h ahead (t+2)
    df[TARGET_COLUMN_72H] = grouped["severe_raw"].shift(-3)  # 72h ahead (t+3)

    # Remover linhas finais com NaN decorrentes do avanço temporal (3 últimos dias de 2025)
    df = df.dropna(subset=[TARGET_COLUMN, TARGET_COLUMN_48H, TARGET_COLUMN_72H]).reset_index(drop=True)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    df[TARGET_COLUMN_48H] = df[TARGET_COLUMN_48H].astype(int)
    df[TARGET_COLUMN_72H] = df[TARGET_COLUMN_72H].astype(int)
    df = df.drop(columns=["severe_raw"])

    # Converter data de volta para string
    df["data"] = df["data"].dt.strftime("%Y-%m-%d")

    # --------------------------------------------------------------------------
    # Organização das Colunas: Isolamento estrito de IDs, Metadados e Features
    # --------------------------------------------------------------------------
    metadata_cols = [
        "cell_id", "data", "ano", "mes", "dia", "dia_do_ano",
        "grid_lat", "grid_lon", "classe_uso_solo", "classe_id", "estacao_mais_proxima",
        # Variáveis contemporâneas de satélite salvas apenas para auditoria (NUNCA usadas como feature)
        "num_focos_satelite", "frp_max_mw", "frp_soma_mw",
        # Alvos futuros
        TARGET_COLUMN, TARGET_COLUMN_48H, TARGET_COLUMN_72H
    ]
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    df_final = df[metadata_cols + feature_cols]

    total_count = len(df_final)
    pos_24h = (df_final[TARGET_COLUMN] == 1).sum()
    pos_48h = (df_final[TARGET_COLUMN_48H] == 1).sum()
    pos_72h = (df_final[TARGET_COLUMN_72H] == 1).sum()

    print("=" * 75)
    print("RESUMO DO DATASET REESTRUTURADO (2016-2025 SEM LEAKAGE):")
    print(f" - Amostras válidas alinhadas: {total_count:,}")
    print(f" - Incidência Alvo 24h (t+1): {pos_24h:,} ({pos_24h/total_count*100:.2f}%)")
    print(f" - Incidência Alvo 48h (t+2): {pos_48h:,} ({pos_48h/total_count*100:.2f}%)")
    print(f" - Incidência Alvo 72h (t+3): {pos_72h:,} ({pos_72h/total_count*100:.2f}%)")
    print(f" - Total de Features Preditivas Limpas: {len(feature_cols)}")
    print(f" - NaNs no Dataset: {df_final.isna().sum().sum()}")
    print("=" * 75)

    # Salvar outputs
    df_final.to_csv(output_final_csv, index=False, encoding="utf-8")
    try:
        df_final.to_parquet(output_final_parquet, index=False)
        print(f"Arquivo Final Parquet salvo em: {output_final_parquet}")
    except Exception:
        pass

    print(f"Arquivo Final CSV salvo em: {output_final_csv}")
    print(">>> [FASE 2.5] Feature Engineering Concluída com Sucesso!\n")
    return df_final


if __name__ == "__main__":
    build_feature_engineering_pipeline()

