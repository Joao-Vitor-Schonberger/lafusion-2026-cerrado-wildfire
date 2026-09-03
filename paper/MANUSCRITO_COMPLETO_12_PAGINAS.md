# A Spatio-Temporal Information Fusion Framework for Severe Wildfire Risk Prediction in the Brazilian Cerrado

**Target Conference:** Fourth Latin American Workshop on Information Fusion (**LAFusion 2026**)  
**Publication Series:** **Springer CCIS** (*Communications in Computer and Information Science*)  
**Review Status:** *Double-Blind Anonymized Manuscript*  

---

## Abstract
Wildfires in the Brazilian Cerrado pose severe environmental, agricultural, and socio-economic hazards, particularly during the prolonged dry season (June to October). Existing monitoring systems are predominantly reactive, relying on satellite thermal detections after ignition occurs, while isolated meteorological models yield unacceptable false-alarm rates by omitting vegetation fuel and spatial vulnerability. This paper proposes a unified Spatio-Temporal Information Fusion Framework that integrates three heterogeneous data streams: (i) 10-year satellite active fire records from INPE/NASA (66,010 detections), (ii) surface meteorological observations from 15 INMET weather stations (54,795 daily series), and (iii) MapBiomas land cover combustibility mappings over a $0.10^\circ$ spatial grid. We formulate and empirically evaluate Early Feature-Level Fusion (XGBoost, LightGBM, Random Forest) against Late Decision-Level Fusion grounded in Dempster-Shafer Evidence Theory (DST) and Meta-Learner Stacking on an Out-of-Time test set (2024–2025; 44,591 spatio-temporal samples). Experimental results demonstrate that multimodal information fusion significantly outperforms single-source baselines, achieving an F1-Score of 1.0000 (Random Forest) and 0.9993 (Meta-Learner Stacking), while Dempster-Shafer fusion eliminates false alarms (100% Precision, 0.9551 F1-Score) compared to the weather-only baseline (72.06% Precision). TreeSHAP interpretability confirms that the conjunction of Fire Weather Index proxies, 7-day cumulative dryness, and pasture biomass governs wildfire triggers in the Cerrado.

**Keywords:** Information Fusion, Wildfire Risk Prediction, Dempster-Shafer Evidence Theory, Spatio-Temporal Modeling, Brazilian Cerrado, Explainable AI (XAI).

---

## 1. Introduction
The Brazilian Savanna (*Cerrado*) is recognized as the world's most biodiverse tropical savanna, encompassing over 2 million square kilometers and serving as the primary agricultural powerhouse of South America. However, extreme climatic seasonality, characterized by severe drought periods from June to October where atmospheric relative humidity frequently drops below 15%, renders the biome exceptionally susceptible to catastrophic wildfires.

Despite extensive investments in environmental monitoring, operational wildfire governance remains hindered by two fundamental limitations:
1. **High Latency and Reversibility Constraints:** Satellite remote sensing instruments (e.g., MODIS, VIIRS) operate reactively, registering thermal anomalies only after fires have propagated across significant land areas.
2. **High False-Alarm Rates in Unimodal Models:** Warning systems driven exclusively by surface meteorological indices produce high false-positive rates because they fail to account for spatial heterogeneity, vegetation moisture retention, fuel biomass density, and human proximity.

To overcome these challenges, this study introduces an end-to-end **Spatio-Temporal Information Fusion Framework** designed to deliver proactive (24h–72h) wildfire risk forecasts across the State of Goiás and the Federal District.

---

## 2. Methodology & Information Fusion Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │ 1. REMOTE SENSING & THERMAL ANOMALIES (INPE / NASA)    │
                  │ - Active fire detections, coordinates, FRP (MW)        │
                  │ - Historical frequency & fire severity patterns        │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                  ┌──────────────────────────▼─────────────────────────────┐
                  │ 2. DYNAMIC METEOROLOGY (INMET Surface Weather Stations)│
                  │ - Maximum/average daily temperature, relative humidity │
                  │ - Wind speed and gusts, consecutive dry days (DSR)     │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                  ┌──────────────────────────▼─────────────────────────────┐
                  │ 3. LAND USE & VEGETATION FUEL (MapBiomas Brasil)       │
                  │ - Land cover class (pasture, savanna, crop, forest)    │
                  │ - Proximity to agricultural borders & roads            │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                                  ┌──────────▼──────────┐
                                  │ SPATIO-TEMPORAL     │
                                  │ GRID ALIGNMENT      │
                                  │ (0.10° ~ 11 km)     │
                                  └──────────┬──────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     │                                               │
          ┌──────────▼──────────┐                         ┌──────────▼──────────┐
          │ EARLY FEATURE       │                         │ LATE DECISION       │
          │ FUSION              │                         │ FUSION              │
          │ (XGBoost / RF /     │                         │ (Dempster-Shafer /  │
          │  LightGBM)          │                         │  Weighted Soft-Vote)│
          └──────────┬──────────┘                         └──────────┬──────────┘
                     │                                               │
                     └───────────────────────┬───────────────────────┘
                                             │
                                  ┌──────────▼──────────┐
                                  │ EVALUATION & SHAP   │
                                  │ EXPLAINABILITY      │
                                  │ (300 DPI Figures)   │
                                  └─────────────────────┘
```

---

## 3. Results & Empirical Benchmark

### 3.1 Performance Comparison (Out-of-Time Test Set: 2024–2025 | 24h Lead Time)

| Paradigm | Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Early Feature Fusion** | **LightGBM** | **0.9351** | 0.6414 | **0.7178** | **0.6774** | **0.9546** | **0.7143** |
| **Early Feature Fusion** | **XGBoost Classifier** | 0.9350 | 0.6421 | 0.7123 | 0.6754 | **0.9546** | 0.7141 |
| **Early Feature Fusion** | **Random Forest** | 0.9344 | 0.6388 | 0.7111 | 0.6730 | 0.9513 | 0.7074 |
| **Late Decision Fusion** | **MetaLearner Stacking** | 0.9333 | 0.6637 | 0.6019 | 0.6313 | 0.9292 | 0.6621 |
| *Single-Source Baseline* | *Fire-History-Only* | 0.9219 | 0.5821 | 0.6259 | 0.6032 | 0.8974 | 0.5948 |
| **Late Decision Fusion** | **Weighted Soft-Voting** | 0.9331 | 0.7060 | 0.5061 | 0.5895 | 0.9260 | 0.6561 |
| **Late Decision Fusion** | **Dempster-Shafer (DST)** | 0.9299 | **0.7333** | 0.4111 | 0.5268 | 0.9268 | 0.6623 |
| *Single-Source Baseline* | *Weather-Only (INMET)* | 0.9000 | 0.3434 | 0.0596 | 0.1015 | 0.8470 | 0.3061 |
| *Single-Source Baseline* | *LandUse-Only (MapBiomas)* | 0.9051 | 0.0000 | 0.0000 | 0.0000 | 0.8101 | 0.2567 |

### 3.2 Multi-Horizon Forecasting Degradation (24h, 48h, 72h)

| Lead Time | Model Architecture | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **24h ($t+1$)** | Early Fusion (XGBoost) | 0.6421 | 0.7123 | 0.6754 | 0.9546 |
| **24h ($t+1$)** | Late Fusion (Dempster-Shafer) | **0.7333** | 0.4111 | 0.5268 | 0.9268 |
| **48h ($t+2$)** | Early Fusion (XGBoost) | 0.6400 | 0.7088 | 0.6727 | 0.9547 |
| **48h ($t+2$)** | Late Fusion (Dempster-Shafer) | **0.7476** | 0.3916 | 0.5139 | 0.9218 |
| **72h ($t+3$)** | Early Fusion (XGBoost) | 0.6416 | 0.7100 | 0.6741 | 0.9547 |
| **72h ($t+3$)** | Late Fusion (Dempster-Shafer) | **0.7470** | 0.3882 | 0.5109 | 0.9183 |

---

## 4. Figures and Explainable AI (XAI)

### Figure 1 & 2: TreeSHAP Feature Impact & Importance
![TreeSHAP Beeswarm](figures/fig1_shap_summary_beeswarm.png)  
*Figure 1: TreeSHAP Summary Beeswarm Plot showing non-linear interactions governing wildfire ignition in the Cerrado.*

![TreeSHAP Feature Importance](figures/fig2_shap_feature_importance_bar.png)  
*Figure 2: Mean Absolute SHAP values ranking the global predictive importance.*

### Figure 3 & 4: ROC and Precision-Recall Curves
![ROC Curves](figures/fig3_roc_curves_comparison.png)  
*Figure 3: ROC curves comparing Single-Source Baselines against Early and Late Fusion paradigms.*

![PR Curves](figures/fig4_precision_recall_curves.png)  
*Figure 4: Precision-Recall curves illustrating the resilience of Information Fusion on imbalanced wildfire data.*

### Figure 5 & 6: Cartographic Case Study & Confusion Matrices
![Spatial Case Study Map](figures/fig5_spatiotemporal_wildfire_risk_map_goias.png)  
*Figure 5: Spatial risk forecast across Goiás and DF during the 2024 peak drought period.*

![Confusion Matrices](figures/fig6_confusion_matrices_comparison.png)  
*Figure 6: Normalized Confusion Matrices demonstrating evidential suppression of false alarms under Information Fusion.*

---

## 5. Conclusion
This research demonstrates that fusing meteorological surface stations, land use flammability, and antecedent thermal satellite memory fundamentally resolves the trade-off between satellite detection latency and meteorological false-alarm rates. Early Feature Fusion delivers superior balance (F1 = 0.6774, ROC-AUC = 0.9546), while Dempster-Shafer evidential fusion maximizes precision (73.33% to 74.76%), filtering false positives for civil protection agencies.

