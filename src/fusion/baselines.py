"""
LAFusion 2026: Spatio-Temporal Information Fusion Framework
Module: src/fusion/baselines.py
Description: Single-source Unimodal Baselines (Weather-only, Fire-history-only, Landuse-only).
Used as benchmarks to quantify the performance gain achieved by multimodal information fusion.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class UnimodalBaselines:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.weather_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1)
        self.fire_history_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1)
        self.landuse_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1)

        self.weather_features = [
            "temp_max_c", "temp_mean_c", "umid_min_pct", "umid_mean_pct",
            "vento_vel_ms", "vento_rajada_ms", "precipitacao_mm", "dias_sem_chuva",
            "umid_min_lag1", "umid_min_lag2", "umid_min_lag3",
            "temp_max_lag1", "temp_max_lag2", "vento_rajada_lag1",
            "umid_min_mean_3d", "umid_min_mean_7d", "temp_max_mean_3d", "temp_max_mean_7d",
            "vento_rajada_max_3d", "precipitacao_acum_7d", "fwi_proxy", "dias_sem_chuva_log"
        ]

        self.fire_history_features = [
            "focos_satelite_lag1", "focos_satelite_lag2", "focos_acum_3d",
            "frp_max_lag1", "frp_soma_lag1"
        ]

        self.landuse_features = [
            "classe_id", "indice_inflamabilidade", "biomassa_combustivel_t_ha",
            "dist_estrada_km", "exposicao_antropica"
        ]

    def fit(self, df_train, target_col="severe_fire_risk_24h"):
        """Train all three single-source baseline models."""
        y_train = df_train[target_col].values

        # 1. Weather-only baseline
        X_weather = df_train[self.weather_features].values
        self.weather_model.fit(X_weather, y_train)

        # 2. Fire-history-only baseline
        X_fire = df_train[self.fire_history_features].values
        self.fire_history_model.fit(X_fire, y_train)

        # 3. Landuse-only baseline
        X_landuse = df_train[self.landuse_features].values
        self.landuse_model.fit(X_landuse, y_train)

        return self

    def predict_proba(self, df_test):
        """Returns predicted probabilities for each single-source baseline."""
        proba_weather = self.weather_model.predict_proba(df_test[self.weather_features].values)[:, 1]
        proba_fire = self.fire_history_model.predict_proba(df_test[self.fire_history_features].values)[:, 1]
        proba_landuse = self.landuse_model.predict_proba(df_test[self.landuse_features].values)[:, 1]

        return {
            "Baseline_WeatherOnly": proba_weather,
            "Baseline_FireHistoryOnly": proba_fire,
            "Baseline_LandUseOnly": proba_landuse,
        }
