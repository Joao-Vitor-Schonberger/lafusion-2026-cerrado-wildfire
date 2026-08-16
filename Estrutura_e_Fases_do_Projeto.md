# Arquitetura e Estrutura de Fases do Projeto — LAFusion 2026

**Documento de Engenharia de Software e Gestão do Projeto**  
**Projeto:** *A Spatio-Temporal Information Fusion Framework for Severe Wildfire Risk Prediction in the Brazilian Cerrado*  
**Evento Alvo:** LAFusion 2026 (*Fourth Latin American Workshop on Information Fusion* — Springer CCIS)  
**Diretório do Projeto:** `C:\Users\joaos\OneDrive\Documentos\2026\01_Academico_e_UFG\Trabalhos\LaFusion`  
**Repositório GitHub:** [https://github.com/Joao-Vitor-Schonberger/lafusion-2026-cerrado-wildfire](https://github.com/Joao-Vitor-Schonberger/lafusion-2026-cerrado-wildfire)  

---

## 🎯 1. Visão Arquitetural do Projeto

O projeto combina **Engenharia de Dados**, **Ciência de Dados (Machine Learning / IA)** e **Metodologia de Publicação Científica** para construir um framework reprodutível e robusto de fusão de dados heterogêneos voltado à predição de queimadas severas no Cerrado.

### 📐 Diagrama de Fluxo e Engenharia:
```
                                 [ FONTES HETEROGÊNEAS ]
                 ┌──────────────────────────┼──────────────────────────┐
                 │                          │                          │
        [ INPE / FIRMS ]             [ INMET Clima ]           [ MapBiomas Terra ]
        66.010 Focos & FRP          54.795 Séries Diárias      5.475 Células de Uso
        (2016-2025 | 10 anos)       (15 Estações de Goiás)     (Cerrado Goiano)
                 │                          │                          │
                 └──────────────────────────┼──────────────────────────┘
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │  FASE 2: ENGENHARIA DE DADOS (CONCLUÍDA)     │
                    │ - Limpeza & Tratamento de Nulos (0 NaNs)     │
                    │ - Grade Espaço-Temporal: 222.833 registros   │
                    │ - 45 Features Multimodais + Target Severo    │
                    └──────────────────────┬───────────────────────┘
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │  FASE 3: MODELAGEM E FUSÃO (CONCLUÍDA)       │
                    │ - 3 Baselines: Clima (72% Prec), Fogo, Solo  │
                    │ - Early Fusion: XGBoost, LightGBM, RF (100%) │
                    │ - Late Fusion: Dempster-Shafer (95.5% F1),   │
                    │   Soft-Voting (99.1% F1), Stacking (99.9% F1)│
                    └──────────────────────┬───────────────────────┘
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │  FASE 4: EXPERIMENTAÇÃO (CONCLUÍDA)          │
                    │ - Fig 1 & 2: SHAP Beeswarm & Importance XAI  │
                    │ - Fig 3 & 4: Curvas ROC & Precision-Recall   │
                    │ - Fig 5 & 6: Mapas Espaciais & Confusão      │
                    │   (300 DPI PNG & PDF vetorial em paper/fig/) │
                    └──────────────────────┬───────────────────────┘
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │  FASE 5: REDAÇÃO E SUBMISSÃO (CONCLUÍDA)     │
                    │ - Template Springer CCIS (LaTeX & Markdown)  │
                    │ - Repositório Oficial Publicado no GitHub    │
                    └──────────────────────┬───────────────────────┘
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │      FASE 6: QA & SUBMISSÃO MICROSOFT CMT    │
                    │ - Auditoria Double-Blind                     │
                    │ - Upload no CMT até 04/09/2026               │
                    └──────────────────────────────────────────────┘
```

---

## 🏁 Tabela Resumo do Cronograma de Fases

| Fase | Foco Principal | Tempo Estimado | Status |
| :--- | :--- | :--- | :--- |
| **Fase 1** | Planejamento, Arquitetura & Setup | Dia 1 | **CONCLUÍDA** ✅ |
| **Fase 2** | Ingestão e Engenharia de Dados (ETL) | Dias 2 a 4 | **CONCLUÍDA** ✅ |
| **Fase 3** | Modelos e Algoritmos de Fusão | Dias 5 a 7 | **CONCLUÍDA** ✅ |
| **Fase 4** | Experimentação, Métricas e SHAP | Dias 8 a 9 | **CONCLUÍDA** ✅ |
| **Fase 5** | Redação do Artigo & Publicação no GitHub | Dias 10 a 14 | **CONCLUÍDA** ✅ |
| **Fase 6** | QA Double-Blind e Submissão CMT | Dias 15 a 16 (Antes de 04/09) | **PRÓXIMO PASSO** ⏳ |
