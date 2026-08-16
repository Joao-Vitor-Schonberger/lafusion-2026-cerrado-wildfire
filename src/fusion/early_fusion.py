"""
LAFusion 2026: Spatio-Temporal Information Fusion Framework
Module: src/fusion/early_fusion.py
Description: Early Feature-Level Multimodal Information Fusion Models (XGBoost, LightGBM, Random Forest).
Fuses all 45 heterogeneous spatial, temporal, meteorological, fuel and satellite features in a joint representation.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class EarlyFeatureFusion:
    def __init__(self, random_state=42):
        self.random_state = random_state

        # Initialize GBDT and Ensemble classifiers for early feature fusion
        self.xgb_model = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )

        self.lgbm_model = LGBMClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.85,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

        self.rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=14,
            random_state=random_state,
            n_jobs=-1,
        )

        self.feature_cols = None

    def fit(self, df_train, target_col="severe_fire_risk", exclude_cols=None):
        """Fit all early fusion classifiers on the joint multimodal feature matrix."""
        if exclude_cols is None:
            exclude_cols = ["cell_id", "data", "classe_uso_solo", "estacao_mais_proxima", target_col]

        self.feature_cols = [c for c in df_train.columns if c not in exclude_cols]
        X_train = df_train[self.feature_cols].values
        y_train = df_train[target_col].values

        print(f"Treinando Early Feature-Level Fusion com {len(self.feature_cols)} features...")
        self.xgb_model.fit(X_train, y_train)
        self.lgbm_model.fit(X_train, y_train)
        self.rf_model.fit(X_train, y_train)
        return self

    def predict_proba(self, df_test):
        """Returns predicted probabilities for each early fusion model."""
        X_test = df_test[self.feature_cols].values

        proba_xgb = self.xgb_model.predict_proba(X_test)[:, 1]
        proba_lgbm = self.lgbm_model.predict_proba(X_test)[:, 1]
        proba_rf = self.rf_model.predict_proba(X_test)[:, 1]

        return {
            "EarlyFusion_XGBoost": proba_xgb,
            "EarlyFusion_LightGBM": proba_lgbm,
            "EarlyFusion_RandomForest": proba_rf,
        }
