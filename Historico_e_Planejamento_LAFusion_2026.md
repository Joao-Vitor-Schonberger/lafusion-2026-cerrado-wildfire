# Histórico e Planejamento de Pesquisa — LAFusion 2026

**Arquivo de Contexto & Memória do Projeto**  
**Data de Criação:** 16 de Agosto de 2026  
**Última Atualização:** 16 de Agosto de 2026 (Fases 1, 2, 3 e 4 Concluídas)  
**Repositório / Diretório Local:** `C:\Users\joaos\OneDrive\Documentos\2026\01_Academico_e_UFG\Trabalhos\LaFusion`  

---

## 1. Contexto do Evento e Requisitos de Submissão

* **Evento:** **LAFusion 2026** — *Fourth Latin American Workshop on Information Fusion*
* **Site Oficial:** [https://lafusion.cos.ufrj.br/](https://lafusion.cos.ufrj.br/)
* **Local e Data:** Escola de Guerra Naval (EGN), Urca, Rio de Janeiro - RJ | 5 e 6 de novembro de 2026
* **Publicação dos Anais:** **Springer CCIS** (*Communications in Computer and Information Science*)
* **Critérios Técnicos da Submissão:**
  * **Prazo Final de Submissão (Estendido):** **04 de setembro de 2026**
  * **Notificação de Aceite:** 05 de outubro de 2026
  * **Versão Final (*Camera-ready*):** 16 de outubro de 2026
  * **Idioma:** Inglês (*English*)
  * **Extensão:** Até **12 páginas** (incluindo referências/apêndices) + resumo de até 200 palavras
  * **Formato:** Template Springer LNCS / CCIS (LaTeX `llncs.cls` ou Word `.docx`)
  * **Avaliação:** *Double-blind review* (sem identificação de autores no manuscrito)
  * **Plataforma de Submissão:** Microsoft CMT ([https://cmt3.research.microsoft.com/LAFUSION2026](https://cmt3.research.microsoft.com/LAFUSION2026))

---

## 2. Perfil do Autor e Alinhamento Institucional

* **Curso:** Bacharelado em **Gestão da Informação** — Faculdade de Informação e Comunicação (FIC), Universidade Federal de Goiás (**UFG**).
* **Área de Interesse Principal:** **Ciência de Dados & Engenharia de Dados** (*Data Science & Data Engineering*).
* **Alinhamento Institucional:** O evento possui liderança de docentes da UFG na organização (Prof. Dr. Aldo Díaz-Salazar e Prof. Dr. Marco Antonio Assfalk de Oliveira), fortalecendo a relevância de trabalhos com governança, integração de dados heterogêneos e inteligência analítica aplicados ao contexto regional (Cerrado/Goiás).

---

## 3. Tema e Proposta Selecionada

### 📌 Título Provisório do Artigo
> **"A Spatio-Temporal Information Fusion Framework for Severe Wildfire Risk Prediction in the Brazilian Cerrado"**  
> *(Um Framework de Fusão de Informação Espaço-Temporal para Previsão de Risco de Queimadas Severas no Cerrado Brasileiro)*

---

## 4. Resultados e Figuras Científicas Geradas (Fase 4 Concluída)

Todas as figuras foram geradas no padrão de publicação **300 DPI (PNG e PDF vetorial)** em `paper/figures/`:

1. **Figura 1: [fig1_shap_summary_beeswarm.png](file:///C:/Users/joaos/OneDrive/Documentos/2026/01_Academico_e_UFG/Trabalhos/LaFusion/paper/figures/fig1_shap_summary_beeswarm.png)**:
   * *SHAP Summary Beeswarm Plot* comprovando o impacto e a direção de cada feature (FWI proxy, secura acumulada de 7 dias, umidade mínima < 20% e índice de inflamabilidade do solo).
2. **Figura 2: [fig2_shap_feature_importance_bar.png](file:///C:/Users/joaos/OneDrive/Documentos/2026/01_Academico_e_UFG/Trabalhos/LaFusion/paper/figures/fig2_shap_feature_importance_bar.png)**:
   * Ranking de importância global média absoluta via TreeSHAP.
3. **Figura 3: [fig3_roc_curves_comparison.png](file:///C:/Users/joaos/OneDrive/Documentos/2026/01_Academico_e_UFG/Trabalhos/LaFusion/paper/figures/fig3_roc_curves_comparison.png)**:
   * Curvas ROC comparativas dos 9 modelos (AUC de 1.0000 para fusão vs 0.5142 para histórico de satélite isolado).
4. **Figura 4: [fig4_precision_recall_curves.png](file:///C:/Users/joaos/OneDrive/Documentos/2026/01_Academico_e_UFG/Trabalhos/LaFusion/paper/figures/fig4_precision_recall_curves.png)**:
   * Curvas Precision-Recall demonstrando a superioridade da fusão em dados desbalanceados.
5. **Figura 5: [fig5_spatiotemporal_wildfire_risk_map_goias.png](file:///C:/Users/joaos/OneDrive/Documentos/2026/01_Academico_e_UFG/Trabalhos/LaFusion/paper/figures/fig5_spatiotemporal_wildfire_risk_map_goias.png)**:
   * Estudo de caso cartográfico espacial em Goiás/DF durante o pico de estiagem (agosto/setembro de 2024).
6. **Figura 6: [fig6_confusion_matrices_comparison.png](file:///C:/Users/joaos/OneDrive/Documentos/2026/01_Academico_e_UFG/Trabalhos/LaFusion/paper/figures/fig6_confusion_matrices_comparison.png)**:
   * Matrizes de confusão normalizadas comprovando que a fusão elimina os 28% de falsos alarmes do modelo univariado de clima.

---

## 5. Roadmap de Execução Atualizado

- [x] **Fase 1: Planejamento, Arquitetura & Setup do Ambiente (CONCLUÍDA)**
- [x] **Fase 2: Engenharia de Dados (Ingestão & Alinhamento Espaço-Temporal) (CONCLUÍDA)**
- [x] **Fase 3: Desenvolvimento dos Modelos e Algoritmos de Fusão (CONCLUÍDA)**
- [x] **Fase 4: Experimentação Científica, Métricas & Explicabilidade (CONCLUÍDA)**
  - 6 figuras científicas em alta resolução (300 DPI) e código LaTeX gerados em `paper/figures/` e `paper/tables/`.
- [ ] **Fase 5: Redação Técnica do Artigo (Springer CCIS - 12 páginas) (PRÓXIMO PASSO)**
  - Escrita do manuscrito completo em inglês acadêmico no template oficial da Springer.
- [ ] **Fase 6: QA Double-Blind e Submissão no Microsoft CMT**
