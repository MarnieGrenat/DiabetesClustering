# Roteiro de Apresentação — Clustering de Pacientes Diabéticos

Tempo total: ~15 minutos. Referências entre parênteses apontam para células do `analysis.ipynb`.

---

## 0. Fluxo de trabalho (~1 min)

> Contar na abertura ou no fechamento — mostra maturidade no processo.

O trabalho foi desenvolvido em duas fases:

**Fase 1 — implementação manual:** Fiz o pré-processamento do dataset (limpeza, encoding, PCA), apliquei K-Means e DBSCAN, e calculei as métricas de avaliação (silhueta, variância intracluster, conectividade). Essa fase deu a base funcional do projeto.

**Fase 2 — refinamento com assistência de IA (vibe coding):** Usei ferramentas de IA para iterar sobre o código e explorar abordagens que eu não conhecia. O ganho mais relevante foi a substituição do PCA pelo **FAMD** (Factor Analysis of Mixed Data) — como 21 das 30 features são categóricas, o PCA sobre dummies one-hot viola a premissa de covariância gaussiana contínua, enquanto o FAMD generaliza PCA + MCA e é projetado para dados mistos. Também refinei o encoding (separar o pipeline de contínuas e categóricas com `ColumnTransformer` em vez de `get_dummies` + `StandardScaler` global), adicionei o teste de Hopkins para validar a clusterabilidade, e incluí o aglomerativo com três linkages.

**O que aprendi com esse fluxo:** A primeira versão funcionava mas tinha decisões ingênuas (PCA em dados categóricos, dummies padronizadas que inflavam categorias raras). O vibe coding não substituiu o entendimento — ele acelerou a descoberta de técnicas que depois precisei entender para justificar. O FAMD, por exemplo, só faz sentido se você souber explicar por que PCA falha em dados mistos.

---

## 1. O Problema (~2 min)

> Célula 0 (título + intro do notebook)

**Abrir com a motivação:**

O dataset *Diabetes 130-US hospitals (1999–2008)* contém 101.766 registros de internações hospitalares de pacientes diabéticos em 130 hospitais americanos, com 50 variáveis clínicas e administrativas. O dataset é classicamente usado para prever readmissão hospitalar (problema supervisionado), mas a AUC típica na literatura fica em ~0.65 — o rótulo é difícil de prever.

Nós invertemos a pergunta: em vez de prever readmissão, queremos **descobrir se existem perfis naturais de pacientes** que emergem dos dados sem usar nenhum rótulo. Isso é útil para:

- Estratificação de risco (quais pacientes precisam de acompanhamento mais intenso?)
- Alocação de recursos (quem se beneficia de home health vs. SNF vs. reabilitação?)
- Descoberta de conhecimento (há subgrupos que modelos preditivos não capturam?)

**Dados-chave para mostrar:**
- 101.766 linhas, 50 colunas (célula 1)
- Variáveis: demográficas (idade, raça, gênero), clínicas (diagnósticos ICD-9, exames, medicações), administrativas (tipo de admissão, destino pós-alta)
- O rótulo `readmitted` existe mas é **guardado como `y_hidden`** — só usado no final para validação externa, nunca como input do clustering

---

## 2. Pré-processamento (~4 min)

> Células 2–26

**Explicar que o preprocessamento foi a parte mais trabalhosa.** Dados clínicos reais são sujos — mais da metade do notebook é limpeza.

### 2.1. Limpeza estrutural (células 2–3)

- Removemos 2.423 linhas de óbito/hospice (`discharge_disposition_id` ∈ {11,13,14,19,20,21}) — pacientes que morreram não podem ser readmitidos, então o `readmitted='NO'` deles é trivial e distorceria os clusters
- Mantivemos apenas a primeira internação por paciente (~30% eram revisitas) — evita violação de independência
- Dropamos `weight` (97% NaN) e `payer_code` (40% NaN, alta cardinalidade)
- Resultado: 69.990 linhas, 48 colunas

### 2.2. IDs administrativos como categóricos (células 4–5)

- `admission_type_id`, `discharge_disposition_id`, `admission_source_id` chegam como inteiros mas são **códigos categóricos** — a "média" de um código não faz sentido
- Mapeamos para buckets clínicos: Emergency/Urgent/Elective, Home/SNF/Rehab/Transfer, Referral/Emergency/Transfer
- **Mostrar a tabela de mapeamento** na célula 5

### 2.3. Agrupamento ICD-9 (células 6–7)

- `diag_1`, `diag_2`, `diag_3` tinham ~700 valores distintos cada — one-hot explodiria em ~2.000 colunas
- Agrupamos em 9 capítulos clínicos (Circulatory, Respiratory, Diabetes, Digestive, etc.)
- **Mostrar a tabela de contagens** da célula 7

### 2.4. NaN como informação (células 8–11)

- A1Cresult (83% NaN) e max_glu_serum (95% NaN) **não são missing-at-random**: o exame só foi pedido quando havia suspeita clínica
- Transformamos NaN → "NotMeasured" para preservar esse sinal
- medical_specialty e race → "Unknown"
- **Mostrar a tabela de informativeness** da célula 9 (delta <5pp, mas preservamos para clusterização)

### 2.5. Drop por variância + transformações (células 12–17)

- Removemos 15 colunas com moda ≥ 99% (medicamentos raríssimos como examide, citoglipton, etc.)
- `log1p()` nas contagens de utilização (number_outpatient, number_emergency, number_inpatient) — distribuição com cauda pesada, sem isso K-Means fica dominado por super-utilizadores
- **Mostrar os histogramas antes/depois** da célula 17

### 2.6. Encoding e redução (células 18–23)

- `ColumnTransformer`: StandardScaler nas 9 contínuas, OneHotEncoder(drop='first') nas 21 categóricas — dummies **não** são re-escaladas (evita peso artificial em categorias raras)
- Resultado: 69.987 × 164 features
- Hopkins H = 0.794 > 0.7 → **dados têm estrutura agrupável**, faz sentido continuar
- **FAMD** (Factor Analysis of Mixed Data) como redução primária → 20 componentes. PCA como fallback
- **Mostrar a projeção 2D FAMD** colorida por `readmitted` (célula 23) — visualização mostra que os rótulos se sobrepõem, confirmando que readmissão não forma clusters naturais

### 2.7. Cenários de outlier (células 24–25)

- IsolationForest(contamination=5%) remove 3.500 outliers → `X_no_iso` (66.487 linhas) para K-Means
- Dataset completo `X_full` preservado para DBSCAN (que identifica outliers como ruído nativamente)
- Amostra estratificada de 5k para hierárquico (limitação O(n²))

---

## 3. Algoritmos de Clustering (~4 min)

> Células 27–39

**Mostrar a tabela resumo do início da Parte II** (célula 27) — três famílias, justificativa de cada substrato.

### 3.1. K-Means++ — varredura de K (células 30–31)

- Sweep K=2..10 no espaço FAMD (20D) sem outliers
- **Mostrar as 4 curvas** (silhueta, DB, CH, SSE) — célula 31
- Melhor K por silhueta: **K=3** (sil=0.1236)
- Tamanhos: 39.322 / 22.291 / 4.874 — partição balanceada e não-trivial
- Comparação K-Means++ vs. random init (célula 33): convergem para a mesma solução (inertia praticamente idêntica) — n_init=10 garante estabilidade da inicialização

### 3.2. DBSCAN (células 34–36)

- Hiperparâmetros: min_samples=10, eps via k-distance plot
- **Mostrar o k-distance plot** com os percentis marcados (célula 35)
- Sweep de eps nos percentis 80/85/90/95:
  - p95 (eps=11.97): 25 clusters, 1.478 ruído, silhueta 0.177
  - Problema: um mega-cluster com 63.350 pontos + 24 micro-clusters periféricos
- **Mostrar a visualização DBSCAN vs. K-Means lado a lado** (célula 36) — evidencia visualmente o mega-cluster e o ruído
- Maldição da dimensionalidade: em 20D as distâncias comprimem, a noção de "vizinhança" perde poder discriminativo

### 3.3. Aglomerativo (células 37–39)

- Single-link, Complete-link, Group average — todos com K=3 na amostra de 5k
- **Mostrar os resultados numéricos** (célula 38):
  - Single: [4994, 2, 2] — chaining clássico, silhueta alta mas artificial
  - Average: [4994, 3, 1] — mesmo problema
  - Complete: [4618, 247, 133] — menos degenerado mas ainda desbalanceado
- **Mostrar os 3 scatter plots lado a lado** (célula 39) — a degeneração é imediatamente visível: um blob gigante com 2–3 pontos coloridos diferente
- Explicar chaining: distância "mínimo entre pares" cria cadeias finas de pontos vizinhos

---

## 4. Avaliação de Resultados (~3 min)

> Células 40–43

### 4.1. Métricas internas (célula 29 + 43)

Explicar brevemente cada métrica e o que significa "melhor":

| Métrica | O que mede | Melhor |
|---------|-----------|--------|
| Silhueta | coesão vs. separação | ↑ (max 1) |
| Davies-Bouldin | dispersão / separação | ↓ (min 0) |
| Calinski-Harabasz | variância between/within | ↑ |
| Variância intracluster (SSE) | soma dos desvios ao centroide | ↓ |
| Conectividade | vizinhos em clusters diferentes | ↓ |

### 4.2. Estabilidade (célula 41)

- 5 bootstraps de 50%, K-Means++ k=3
- ARI médio entre bootstraps = 0.473
- Interpretação: estabilidade **moderada** — a estrutura existe mas é sensível à composição da amostra
- Por que não validação cruzada? Não existe "k-fold" para clustering não-supervisionado. Bootstrap é o análogo correto

### 4.3. Tabela comparativa (célula 43)

**Mostrar a tabela completa e o ranking por silhueta.**

Ponto crucial da apresentação — explicar o paradoxo:

- Aglomerativo single/average tem silhueta 0.37–0.40 (a maior!) mas com clusters **degenerados** [4994, 2, 2]
- K-Means tem silhueta 0.12 (a menor) mas partições **balanceadas e interpretáveis**
- Lição: **silhueta alta ≠ clustering útil**. Uma métrica sozinha não decide — é preciso olhar o formato dos clusters

### 4.4. Validação externa (célula 43)

- ARI ≈ 0 contra `readmitted` para todos os algoritmos
- Isso **não é falha** — readmissão tem componente comportamental/social que features clínicas não capturam
- A literatura supervisionada reporta AUC ~0.65 no mesmo dataset, ou seja, mesmo modelos treinados explicitamente para o rótulo não vão muito além do acaso

---

## 5. Conclusão (~2 min)

> Células 44–46

### 5.1. Interpretação clínica (célula 45)

**Mostrar as tabelas de caracterização** e descrever os três perfis:

- **Cluster 0** (59%): *Clínico-laboratorial* — muitos exames (47.2), poucos procedimentos (1.03), diabetes como diag primário em 9.8%, 15% alta para SNF, readmissão 36%
- **Cluster 1** (34%): *Cirúrgico/eletivo* — muitos procedimentos (2.05) e medicações (17.2), diagnósticos musculoesqueléticos (12.4%) e neoplasias (8.1%), 67% alta para casa, readmissão 30%
- **Cluster 2** (7%): *Crônico de alta utilização* — maior histórico de visitas prévias, 3.2% transferência hospitalar, readmissão 41%

**Mostrar o scatter plot FAMD 2D** colorido por cluster (célula 45).

### 5.2. Limitações

Métodos:
- K-Means assume esfericidade — dummies binárias não são naturalmente esféricas
- DBSCAN falha em 20D (maldição da dimensionalidade)
- Aglomerativo limitado a 5k pontos (O(n²))

Métricas:
- Silhueta favorece convexidade — em dados com sobreposição natural, valores baixos são esperados
- Davies-Bouldin enviesado pela mesma geometria do K-Means
- Conectividade depende da subamostra

### 5.3. Veredito

- Resultado: **parcialmente satisfatório** — silhueta modesta (0.12) mas clusters clinicamente coerentes
- Melhor algoritmo: **K-Means++ k=3** — único com partições não-degeneradas e traduzíveis em ações hospitalares
- Os agrupamentos **fazem sentido clínico**: clínico vs. cirúrgico vs. crônico é uma distinção reconhecida na gestão hospitalar

---

## Dicas para a apresentação

- **Células para mostrar rodando**: 1 (shape), 7 (ICD-9), 17 (histogramas), 21 (Hopkins), 23 (FAMD 2D), 31 (curvas K-Means), 35 (k-distance), **36 (DBSCAN vs K-Means 2D)**, 38 (hierárquico números), **39 (hierárquico 3 scatter plots)**, 43 (tabela final), 45 (caracterização + scatter)
- **Não tente mostrar todas as células** — foque nas de output visual (gráficos e tabelas)
- **Prepare-se para a pergunta "por que silhueta tão baixa?"** — a resposta é que dados clínicos tabulares formam um espectro, não ilhas, e 0.12 é consistente com a literatura
- **Prepare-se para "por que não Ward linkage?"** — Ward é equivalente a minimizar inertia (como K-Means), o que o torna redundante na comparação; priorizamos linkages que capturam geometrias diferentes
- **Prepare-se para "por que FAMD e não PCA?"** — 21 de 30 features são categóricas; PCA sobre one-hot dummies viola a premissa de covariância gaussiana contínua; FAMD generaliza PCA + MCA para dados mistos
