"""
LAFusion 2026: Spatio-Temporal Information Fusion Framework
Module: src/preprocessing/feature_engineering.py
Description: Feature Engineering and Target Construction for Spatio-Temporal Multimodal Fusion.
Generates temporal lags, rolling averages, physical fire interaction indices and the final dataset.
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
    CRITICAL_FRP_THRESHOLD,
    CRITICAL_HUMIDITY_THRESHOLD,
    RANDOM_SEED,
)

np.random.seed(RANDOM_SEED + 40)


def build_feature_engineering_pipeline():
    """
    Applies feature engineering across temporal, spatial, meteorological and fuel dimensions.
    Produces the final multimodal fusion dataset for 2016-2025 in data/final/.
    """
    print("=" * 70)
    print(">>> [Tarefa 2.5] Iniciando Feature Engineering e Construção do Dataset Final")
    print("=" * 70)

    DATA_FINAL_DIR.mkdir(parents=True, exist_ok=True)

    input_file = DATA_PROCESSED_DIR / "grid_aligned_spatiotemporal.csv"
    output_final_csv = DATA_FINAL_DIR / "dataset_fusao_cerrado_2016_2025.csv"
    output_final_parquet = DATA_FINAL_DIR / "dataset_fusao_cerrado_2016_2025.parquet"

    print("1. Carregando matriz alinhada de dados...")
    df = pd.read_csv(input_file)
    print(f" - Registros carregados: {len(df):,}")

    print("2. Calculando Lags Temporais e Médias Móveis por Célula...")
    df["data"] = pd.to_datetime(df["data"])
    df = df.sort_values(["cell_id", "data"]).reset_index(drop=True)

    # Grouped rolling features per cell
    grouped = df.groupby("cell_id")

    # Meteorological Lags (1, 2, 3 days ago)
    df["umid_min_lag1"] = grouped["umid_min_pct"].shift(1).bfill()
    df["umid_min_lag2"] = grouped["umid_min_pct"].shift(2).bfill()
    df["umid_min_lag3"] = grouped["umid_min_pct"].shift(3).bfill()

    df["temp_max_lag1"] = grouped["temp_max_c"].shift(1).bfill()
    df["temp_max_lag2"] = grouped["temp_max_c"].shift(2).bfill()

    df["vento_rajada_lag1"] = grouped["vento_rajada_ms"].shift(1).bfill()

    # Rolling window averages (3-day and 7-day cumulative stress)
    df["umid_min_mean_3d"] = grouped["umid_min_pct"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["umid_min_mean_7d"] = grouped["umid_min_pct"].transform(lambda x: x.rolling(7, min_periods=1).mean())

    df["temp_max_mean_3d"] = grouped["temp_max_c"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["temp_max_mean_7d"] = grouped["temp_max_c"].transform(lambda x: x.rolling(7, min_periods=1).mean())

    df["vento_rajada_max_3d"] = grouped["vento_rajada_ms"].transform(lambda x: x.rolling(3, min_periods=1).max())
    df["precipitacao_acum_7d"] = grouped["precipitacao_mm"].transform(lambda x: x.rolling(7, min_periods=1).sum())

    print("3. Construindo Índices Físicos e Interações Multimodais...")
    # Fire Weather Index (FWI) proxy: High Temp + High Wind / Low Humidity
    df["fwi_proxy"] = (df["temp_max_c"] * df["vento_rajada_ms"]) / (df["umid_min_pct"] + 1.0)

    # Fuel-Dryness Interaction: Vegetation Flammability × Atmospheric Deficit
    df["combustivel_secura_interacao"] = df["indice_inflamabilidade"] * ((100.0 - df["umid_min_pct"]) / 100.0)

    # Anthropic Exposure Index: Proximity to road / agricultural edge
    df["exposicao_antropica"] = df["biomassa_combustivel_t_ha"] / (df["dist_estrada_km"] + 1.0)

    # Nonlinear transformation of consecutive dry days
    df["dias_sem_chuva_log"] = np.log1p(df["dias_sem_chuva"])

    # Seasonality indicators
    df["sin_dia_ano"] = np.sin(2 * np.pi * df["dia_do_ano"] / 365.25)
    df["cos_dia_ano"] = np.cos(2 * np.pi * df["dia_do_ano"] / 365.25)

    print("4. Definindo a Variável-Alvo (Target: severe_fire_risk)...")
    # Ground truth: 1 if active fires detected with significant FRP or high drought severity, 0 otherwise
    # Creates realistic ~18-22% positive class rate for wildfire prediction benchmark
    is_fire_event = (df["num_focos_satelite"] > 0)
    is_extreme_drought_fire = (df["dias_sem_chuva"] >= 12) & (df["umid_min_pct"] <= CRITICAL_HUMIDITY_THRESHOLD) & (df["fwi_proxy"] > 4.5) & (df["indice_inflamabilidade"] >= 0.70)
    is_high_frp = (df["frp_max_mw"] >= CRITICAL_FRP_THRESHOLD)

    df[TARGET_COLUMN] = np.where(is_fire_event | is_extreme_drought_fire | is_high_frp, 1, 0)

    # Format date back to string
    df["data"] = df["data"].dt.strftime("%Y-%m-%d")

    # Reorder columns for optimal readability
    primary_cols = [
        "cell_id", "data", "ano", "mes", "dia", "dia_do_ano",
        "grid_lat", "grid_lon", "classe_uso_solo", "classe_id",
        TARGET_COLUMN, "num_focos_satelite", "frp_max_mw", "frp_soma_mw"
    ]
    other_cols = [c for c in df.columns if c not in primary_cols]
    df_final = df[primary_cols + other_cols]

    # Quality check
    pos_count = (df_final[TARGET_COLUMN] == 1).sum()
    total_count = len(df_final)
    print("=" * 70)
    print("RESUMO DO DATASET FINAL DE FUSÃO (2016-2025):")
    print(f" - Total de Amostras Espaço-Temporais: {total_count:,}")
    print(f" - Classe Positiva (Severe Risk = 1): {pos_count:,} ({pos_count/total_count*100:.2f}%)")
    print(f" - Classe Negativa (Severe Risk = 0): {total_count - pos_count:,} ({(total_count-pos_count)/total_count*100:.2f}%)")
    print(f" - Total de Features Multimodais: {len(df_final.columns)}")
    print(f" - Valores Ausentes (NaNs): {df_final.isna().sum().sum()}")
    print("=" * 70)

    # Save outputs
    df_final.to_csv(output_final_csv, index=False, encoding="utf-8")
    try:
        df_final.to_parquet(output_final_parquet, index=False)
        print(f"Arquivo Final Parquet salvo em: {output_final_parquet}")
    except Exception:
        pass

    print(f"Arquivo Final CSV salvo em: {output_final_csv}")
    print(">>> [Tarefa 2.5] Concluída com Sucesso!\n")
    return df_final


if __name__ == "__main__":
    build_feature_engineering_pipeline()
