"""
LAFusion 2026: Spatio-Temporal Information Fusion Framework
Module: src/fusion/late_fusion.py
Description: Late Decision-Level Information Fusion (Dempster-Shafer Evidence Theory, Weighted Soft-Voting, Meta-Learner Stacking).
Combines independent domain expert classifiers handling uncertainty and conflicting evidence.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def dempster_shafer_combine_two(m1, m2):
    """
    Combines two Basic Belief Assignments (BBAs) over Omega = {Fire, NoFire}
    m = (m_fire, m_nofire, m_omega)
    """
    m1_f, m1_nf, m1_om = m1
    m2_f, m2_nf, m2_om = m2

    # Conflict metric K
    K = (m1_f * m2_nf) + (m1_nf * m2_f)
    if K >= 1.0 - 1e-7:
        K = 1.0 - 1e-7  # Avoid division by zero in total conflict

    denom = 1.0 - K

    # Orthogonal combination sum
    m_comb_f = ((m1_f * m2_f) + (m1_f * m2_om) + (m1_om * m2_f)) / denom
    m_comb_nf = ((m1_nf * m2_nf) + (m1_nf * m2_om) + (m1_om * m2_nf)) / denom
    m_comb_om = (m1_om * m2_om) / denom

    return m_comb_f, m_comb_nf, m_comb_om


class LateDecisionFusion:
    def __init__(self, random_state=42):
        self.random_state = random_state

        # Domain Expert 1: Meteorology Specialist
        self.expert_weather = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.08, random_state=random_state, n_jobs=-1, eval_metric="logloss"
        )
        # Domain Expert 2: Land Cover & Fuel Specialist
        self.expert_landuse = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1
        )
        # Domain Expert 3: Satellite Thermal Anomalies Specialist
        self.expert_satellite = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.08, random_state=random_state, n_jobs=-1, eval_metric="logloss"
        )

        # Meta-Learner Stacking
        self.meta_learner = LogisticRegression(random_state=random_state)

        # Feature sets per expert
        self.weather_features = [
            "temp_max_c", "temp_mean_c", "umid_min_pct", "umid_mean_pct",
            "vento_vel_ms", "vento_rajada_ms", "precipitacao_mm", "dias_sem_chuva",
            "umid_min_mean_3d", "umid_min_mean_7d", "temp_max_mean_7d", "vento_rajada_max_3d",
            "fwi_proxy", "dias_sem_chuva_log"
        ]

        self.landuse_features = [
            "classe_id", "indice_inflamabilidade", "biomassa_combustivel_t_ha",
            "dist_estrada_km", "combustivel_secura_interacao", "exposicao_antropica"
        ]

        self.satellite_features = [
            "focos_satelite_lag1", "focos_satelite_lag2", "focos_acum_3d",
            "frp_max_lag1", "frp_soma_lag1"
        ]

    def fit(self, df_train, target_col="severe_fire_risk_24h"):
        """Train the 3 domain specialists and fit the Meta-Learner."""
        y_train = df_train[target_col].values

        print("Treinando Especialistas de Domínio (Clima, Solo, Satélite)...")
        self.expert_weather.fit(df_train[self.weather_features].values, y_train)
        self.expert_landuse.fit(df_train[self.landuse_features].values, y_train)
        self.expert_satellite.fit(df_train[self.satellite_features].values, y_train)

        # Get in-sample specialist probabilities for meta-learner training
        p_w = self.expert_weather.predict_proba(df_train[self.weather_features].values)[:, 1]
        p_l = self.expert_landuse.predict_proba(df_train[self.landuse_features].values)[:, 1]
        p_s = self.expert_satellite.predict_proba(df_train[self.satellite_features].values)[:, 1]

        X_meta_train = np.column_stack([p_w, p_l, p_s])
        self.meta_learner.fit(X_meta_train, y_train)
        return self

    def predict_proba(self, df_test, horizon=1):
        """
        Generates fused decision probabilities via Soft-Voting, Dempster-Shafer (DST), and Stacking.
        DST applies dynamic evidential discounting based on physical distance to weather stations
        and temporal lag decay.
        """
        p_w = self.expert_weather.predict_proba(df_test[self.weather_features].values)[:, 1]
        p_l = self.expert_landuse.predict_proba(df_test[self.landuse_features].values)[:, 1]
        p_s = self.expert_satellite.predict_proba(df_test[self.satellite_features].values)[:, 1]

        # 1. Weighted Soft-Voting Fusion (heuristic weights: 0.45 weather, 0.25 landuse, 0.30 satellite)
        proba_soft_voting = (0.45 * p_w) + (0.25 * p_l) + (0.30 * p_s)

        # 2. Meta-Learner Stacking Fusion
        X_meta_test = np.column_stack([p_w, p_l, p_s])
        proba_stacking = self.meta_learner.predict_proba(X_meta_test)[:, 1]

        # 3. Dynamic Dempster-Shafer Evidence Theory Fusion (DST)
        # Spatial discounting: weather confidence decays with distance from physical station
        if "dist_estacao_km" in df_test.columns:
            dist_km = df_test["dist_estacao_km"].values
            alpha_w_dynamic = np.clip(0.92 * np.exp(-dist_km / 150.0), 0.50, 0.95)
        else:
            alpha_w_dynamic = np.full(len(df_test), 0.88)

        # Temporal discounting for satellite memory across horizons
        alpha_s_dynamic = np.full(len(df_test), max(0.65, 0.90 * np.exp(-0.20 * (horizon - 1))))
        alpha_l = 0.82

        dst_probs = []
        for pw_i, pl_i, ps_i, aw_i, as_i in zip(p_w, p_l, p_s, alpha_w_dynamic, alpha_s_dynamic):
            # Form BBAs for each source
            m_w = (aw_i * pw_i, aw_i * (1.0 - pw_i), 1.0 - aw_i)
            m_l = (alpha_l * pl_i, alpha_l * (1.0 - pl_i), 1.0 - alpha_l)
            m_s = (as_i * ps_i, as_i * (1.0 - ps_i), 1.0 - as_i)

            # Combine source 1 and 2
            m_wl = dempster_shafer_combine_two(m_w, m_l)
            # Combine result with source 3
            m_final = dempster_shafer_combine_two(m_wl, m_s)

            # Pignistic transformation: BetP(Fire) = m(Fire) + 0.5 * m(Omega)
            bet_p_fire = m_final[0] + 0.5 * m_final[2]
            dst_probs.append(bet_p_fire)

        proba_dempster_shafer = np.array(dst_probs)

        return {
            "LateFusion_WeightedSoftVoting": proba_soft_voting,
            "LateFusion_MetaLearnerStacking": proba_stacking,
            "LateFusion_DempsterShafer": proba_dempster_shafer,
        }

