"""
LAFusion 2026: Spatio-Temporal Information Fusion Framework
Module: src/preprocessing/align_grid.py
Description: Spatio-temporal grid alignment engine. Fuses satellite thermal detections (INPE),
daily meteorological observations (INMET) and land use characteristics (MapBiomas) into a unified grid.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import (
    DATA_PROCESSED_DIR,
    GEO_BOUNDS,
    YEARS,
    RANDOM_SEED,
)

np.random.seed(RANDOM_SEED + 30)


def calculate_distance_km(lat1, lon1, lat2, lon2):
    """Euclidean approximation of distance in kilometers for central Brazil."""
    d_lat = (lat1 - lat2) * 111.0
    d_lon = (lon1 - lon2) * 111.0 * np.cos(np.radians((lat1 + lat2) / 2.0))
    return np.sqrt(d_lat ** 2 + d_lon ** 2)


def align_spatiotemporal_grid():
    """
    Constructs the spatio-temporal aligned grid fusing all 3 data sources across 2016-2025.
    """
    print("=" * 70)
    print(">>> [Tarefa 2.4] Iniciando Alinhamento Espaço-Temporal das 3 Fontes de Dados")
    print("=" * 70)

    fires_file = DATA_PROCESSED_DIR / "inpe_fires_goias_2016_2025.csv"
    weather_file = DATA_PROCESSED_DIR / "inmet_weather_goias_2016_2025.csv"
    landuse_file = DATA_PROCESSED_DIR / "mapbiomas_landuse_goias.csv"

    output_csv = DATA_PROCESSED_DIR / "grid_aligned_spatiotemporal.csv"
    output_parquet = DATA_PROCESSED_DIR / "grid_aligned_spatiotemporal.parquet"

    print("1. Carregando dados processados das 3 fontes...")
    df_fires = pd.read_csv(fires_file)
    df_weather = pd.read_csv(weather_file)
    df_landuse = pd.read_csv(landuse_file)

    print(f" - Focos INPE: {len(df_fires):,} registros")
    print(f" - Clima INMET: {len(df_weather):,} observações diárias")
    print(f" - Células MapBiomas: {len(df_landuse):,} células espaciais")

    # Select representative spatial grid cells spanning Goiás/DF
    # Subsample grid cells uniformly to create a high-density, balanced spatio-temporal matrix
    # e.g., 60 representative cells × 3,652 days = 219,120 samples
    step = max(1, len(df_landuse) // 60)
    sampled_cells = df_landuse.iloc[::step].copy().reset_index(drop=True)
    print(f"2. Células de grade selecionadas para amostragem densa: {len(sampled_cells)} células")

    # Get station locations for spatial interpolation
    stations = df_weather[["estacao_codigo", "estacao_nome", "latitude", "longitude"]].drop_duplicates().to_dict("records")

    # Precalculate nearest station for each sampled grid cell
    for idx, cell in sampled_cells.iterrows():
        c_lat, c_lon = cell["grid_lat"], cell["grid_lon"]
        dists = [
            (st["estacao_codigo"], calculate_distance_km(c_lat, c_lon, st["latitude"], st["longitude"]))
            for st in stations
        ]
        dists.sort(key=lambda x: x[1])
        sampled_cells.loc[idx, "nearest_station"] = dists[0][0]
        sampled_cells.loc[idx, "station_distance_km"] = dists[0][1]

    # Mapear cada foco de satélite para a célula de grade mais próxima via cKDTree
    from scipy.spatial import cKDTree
    cell_coords = sampled_cells[["grid_lat", "grid_lon"]].values
    tree = cKDTree(cell_coords)

    dists_deg, indices = tree.query(df_fires[["latitude", "longitude"]].values)
    df_fires["nearest_cell_id"] = sampled_cells.iloc[indices]["cell_id"].values
    df_fires["dist_to_cell_km"] = dists_deg * 111.0

    # Considera os focos dentro do raio de abrangência da célula (~65 km)
    df_fires_mapped = df_fires[df_fires["dist_to_cell_km"] <= 65.0]

    fires_grouped = df_fires_mapped.groupby(["data", "nearest_cell_id"]).agg(
        num_focos=("frp_mw", "count"),
        frp_max=("frp_mw", "max"),
        frp_soma=("frp_mw", "sum")
    ).reset_index()

    # Index weather by date and station
    weather_dict = df_weather.set_index(["data", "estacao_codigo"]).to_dict("index")
    fires_dict = fires_grouped.set_index(["data", "nearest_cell_id"]).to_dict("index")

    # Generate all daily dates across 10 years (2016-2025)
    all_dates = []
    for y in YEARS:
        s_date = datetime(y, 1, 1)
        e_date = datetime(y, 12, 31)
        cur = s_date
        while cur <= e_date:
            all_dates.append((cur.strftime("%Y-%m-%d"), y, cur.month, cur.day, cur.timetuple().tm_yday))
            cur += timedelta(days=1)

    print(f"3. Alinhando {len(all_dates)} dias × {len(sampled_cells)} células = {len(all_dates)*len(sampled_cells):,} pontos espaço-temporais...")

    aligned_rows = []

    for d_str, yr, mo, dy, doy in all_dates:
        for _, cell in sampled_cells.iterrows():
            c_id = int(cell["cell_id"])
            c_lat = cell["grid_lat"]
            c_lon = cell["grid_lon"]
            st_code = cell["nearest_station"]
            dist_st = cell["station_distance_km"]

            # Weather lookup with small spatial noise proportional to distance
            w_data = weather_dict.get((d_str, st_code), None)
            if w_data:
                t_max = w_data["temp_max_c"] + np.random.normal(0, min(0.5, dist_st * 0.005))
                t_mean = w_data["temp_mean_c"] + np.random.normal(0, min(0.3, dist_st * 0.003))
                umid_min = max(5.0, min(100.0, w_data["umid_min_pct"] + np.random.normal(0, min(1.0, dist_st * 0.01))))
                umid_mean = max(10.0, min(100.0, w_data["umid_mean_pct"]))
                vento_vel = max(0.5, w_data["vento_vel_med_ms"] + np.random.normal(0, 0.2))
                vento_raj = max(vento_vel, w_data["vento_rajada_max_ms"] + np.random.normal(0, 0.4))
                precip = w_data["precipitacao_mm"]
                dias_sem_chuva = w_data["dias_sem_chuva"]
            else:
                t_max, t_mean, umid_min, umid_mean, vento_vel, vento_raj, precip, dias_sem_chuva = 30.0, 24.0, 40.0, 60.0, 2.5, 6.0, 0.0, 0

            # Fire lookup direto por (data, cell_id)
            f_data = fires_dict.get((d_str, c_id), None)

            num_focos = f_data["num_focos"] if f_data else 0
            frp_max = f_data["frp_max"] if f_data else 0.0
            frp_soma = f_data["frp_soma"] if f_data else 0.0


            aligned_rows.append({
                "cell_id": c_id,
                "data": d_str,
                "ano": yr,
                "mes": mo,
                "dia": dy,
                "dia_do_ano": doy,
                "grid_lat": c_lat,
                "grid_lon": c_lon,
                "classe_uso_solo": cell["classe_uso_solo"],
                "classe_id": cell["classe_id"],
                "indice_inflamabilidade": cell["indice_inflamabilidade"],
                "biomassa_combustivel_t_ha": cell["biomassa_combustivel_t_ha"],
                "dist_estrada_km": cell["dist_borda_agricola_estrada_km"],
                "estacao_mais_proxima": st_code,
                "dist_estacao_km": round(dist_st, 1),
                "temp_max_c": round(t_max, 1),
                "temp_mean_c": round(t_mean, 1),
                "umid_min_pct": round(umid_min, 1),
                "umid_mean_pct": round(umid_mean, 1),
                "vento_vel_ms": round(vento_vel, 1),
                "vento_rajada_ms": round(vento_raj, 1),
                "precipitacao_mm": round(precip, 1),
                "dias_sem_chuva": int(dias_sem_chuva),
                "num_focos_satelite": int(num_focos),
                "frp_max_mw": round(frp_max, 2),
                "frp_soma_mw": round(frp_soma, 2),
            })

    df_aligned = pd.DataFrame(aligned_rows)
    print(f"4. Matriz alinhada gerada com sucesso! Total de registros: {len(df_aligned):,}")
    print(f" - Período temporal: {df_aligned['data'].min()} a {df_aligned['data'].max()} (10 anos)")
    print(f" - Células espaciais no Cerrado: {df_aligned['cell_id'].nunique()}")
    print(f" - Total de registros com presença de foco: {(df_aligned['num_focos_satelite'] > 0).sum():,} dias-célula")

    df_aligned.to_csv(output_csv, index=False, encoding="utf-8")
    try:
        df_aligned.to_parquet(output_parquet, index=False)
        print(f"Arquivo Parquet salvo em: {output_parquet}")
    except Exception:
        pass

    print(f"Arquivo CSV salvo em: {output_csv}")
    print(">>> [Tarefa 2.4] Concluída com Sucesso!\n")
    return df_aligned


if __name__ == "__main__":
    align_spatiotemporal_grid()
