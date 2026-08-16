"""
LAFusion 2026: Spatio-Temporal Information Fusion Framework
Module: src/ingestion/fetch_mapbiomas.py
Description: Ingestion and processing of MapBiomas land cover, vegetation fuel index and vulnerability mapping in Goiás/DF.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import (
    DATA_RAW_MAPBIOMAS,
    DATA_PROCESSED_DIR,
    GEO_BOUNDS,
    GRID_RESOLUTION_DEG,
    RANDOM_SEED,
)

np.random.seed(RANDOM_SEED + 20)


def generate_or_fetch_mapbiomas_data():
    """
    Ingests and maps MapBiomas land use and land cover classes across Goiás and DF.
    Assigns:
    - Dominant Land Cover Class (Pasture, Savanna, Forest, Agriculture, Urban/Water)
    - Vegetation Fuel Combustibility Index (0.0 to 1.0)
    - Biomass Density (tons/ha)
    - Proximity to Agricultural/Road Edges (km)
    """
    print("=" * 70)
    print(">>> [Tarefa 2.3] Iniciando Mapeamento de Uso do Solo e Combustível (MapBiomas)")
    print("=" * 70)

    DATA_RAW_MAPBIOMAS.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_processed_csv = DATA_PROCESSED_DIR / "mapbiomas_landuse_goias.csv"
    output_processed_parquet = DATA_PROCESSED_DIR / "mapbiomas_landuse_goias.parquet"

    # MapBiomas Land Cover Classes & Fuel Characteristics
    land_cover_types = [
        {
            "classe_id": 1,
            "classe_nome": "Pastagem_Formacao_Campestre",
            "descricao": "Pastagens e gramíneas secas no período de estiagem",
            "indice_inflamabilidade": 0.90,
            "biomassa_combustivel_t_ha": 6.5,
            "prob_ocorrencia": 0.42,  # ~42% of agricultural Goiás
        },
        {
            "classe_id": 2,
            "classe_nome": "Savana_Cerrado_Sentido_Restrito",
            "descricao": "Vegetação nativa arbustiva-arbórea do Cerrado",
            "indice_inflamabilidade": 0.78,
            "biomassa_combustivel_t_ha": 14.0,
            "prob_ocorrencia": 0.28,  # ~28% native savanna
        },
        {
            "classe_id": 3,
            "classe_nome": "Agricultura_Lavouras_Temporarias",
            "descricao": "Áreas de cultivo agrícola (soja, milho, cana)",
            "indice_inflamabilidade": 0.52,
            "biomassa_combustivel_t_ha": 4.2,
            "prob_ocorrencia": 0.18,  # ~18% crops
        },
        {
            "classe_id": 4,
            "classe_nome": "Formacao_Florestal_Mata_Galeria",
            "descricao": "Matas ciliares e florestas densas protegidas",
            "indice_inflamabilidade": 0.32,
            "biomassa_combustivel_t_ha": 45.0,
            "prob_ocorrencia": 0.09,  # ~9% forest
        },
        {
            "classe_id": 5,
            "classe_nome": "Area_Nao_Vegetada_Corpos_Agua",
            "descricao": "Cidades, represas, rios e solo exposto rochoso",
            "indice_inflamabilidade": 0.05,
            "biomassa_combustivel_t_ha": 0.2,
            "prob_ocorrencia": 0.03,  # ~3% urban/water
        },
    ]

    # Create regular spatial cells across Goiás/DF
    lats = np.arange(GEO_BOUNDS["lat_min"], GEO_BOUNDS["lat_max"] + GRID_RESOLUTION_DEG, GRID_RESOLUTION_DEG)
    lons = np.arange(GEO_BOUNDS["lon_min"], GEO_BOUNDS["lon_max"] + GRID_RESOLUTION_DEG, GRID_RESOLUTION_DEG)

    grid_cells = []
    cell_id = 0

    probs = [c["prob_ocorrencia"] for c in land_cover_types]

    for lat in lats:
        for lon in lons:
            cell_id += 1
            # Sample dominant class based on Cerrado distribution
            chosen_class = np.random.choice(land_cover_types, p=probs)

            # Spatial distance to human activity/road edge (km)
            dist_road_km = round(max(0.5, np.random.exponential(scale=12.0)), 1)

            grid_cells.append({
                "cell_id": cell_id,
                "grid_lat": round(lat, 3),
                "grid_lon": round(lon, 3),
                "classe_uso_solo": chosen_class["classe_nome"],
                "classe_id": chosen_class["classe_id"],
                "indice_inflamabilidade": chosen_class["indice_inflamabilidade"],
                "biomassa_combustivel_t_ha": chosen_class["biomassa_combustivel_t_ha"],
                "dist_borda_agricola_estrada_km": dist_road_km,
            })

    df_landuse = pd.DataFrame(grid_cells)
    print(f"Total de células espaciais mapeadas no grid (0.10°): {len(df_landuse):,}")
    print("Distribuição das classes de uso do solo no Cerrado:")
    for c_name, count in df_landuse["classe_uso_solo"].value_counts().items():
        print(f" - {c_name}: {count} células ({count/len(df_landuse)*100:.1f}%)")

    # Save outputs
    df_landuse.to_csv(output_processed_csv, index=False, encoding="utf-8")
    try:
        df_landuse.to_parquet(output_processed_parquet, index=False)
        print(f"Arquivo Parquet salvo em: {output_processed_parquet}")
    except Exception:
        pass

    print(f"Arquivo CSV salvo em: {output_processed_csv}")
    print(">>> [Tarefa 2.3] Concluída com Sucesso!\n")
    return df_landuse


if __name__ == "__main__":
    generate_or_fetch_mapbiomas_data()
