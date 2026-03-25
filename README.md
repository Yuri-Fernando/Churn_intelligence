# Churn Intelligence Pipeline

Projeto de pesquisa aplicada em Inteligencia Artificial que simula um **Customer Data Platform (CDP)** moderno. O sistema recebe dados de comportamento de usuarios, prediz risco de churn, segmenta a base, toma decisoes automaticas de retencao com agentes cognitivos e expoe tudo via API — com rastreabilidade, explicabilidade e analise de fairness.

---

## O que esse projeto faz

O problema central e: **como saber, antes de acontecer, que um usuario vai parar de usar o servico — e o que fazer a respeito?**

O pipeline responde isso em camadas:

**1. Processa dados brutos de comportamento**
Recebe um dataset de 50.000 usuarios de e-commerce com informacoes de engajamento, compras e sessoes. Constroi features comportamentais (recencia, frequencia, intensidade de uso) que capturam padroes de comportamento ao longo do tempo — nao apenas atributos estaticos.

**2. Prediz churn com Machine Learning**
Treina tres modelos (Logistic Regression, Random Forest, Gradient Boosting) para calcular um `churn_score` entre 0 e 1 para cada usuario. Usa SHAP para explicar quais features mais influenciam cada predicao.

**3. Segmenta usuarios dinamicamente**
Agrupa usuarios em 4 segmentos via KMeans (engaged, occasional, at_risk, inactive) com base no comportamento real — nao em regras fixas. Cada segmento recebe uma acao de retencao associada.

**4. Toma decisoes com agentes cognitivos**
Usuarios na "zona de incerteza" (score entre 0.4 e 0.7) passam por um pipeline de agentes especializados: o Analyst interpreta o score, o Strategy decide a acao, o RAG gera uma explicacao narrativa e o Auditor valida a consistencia da decisao.

**5. Explica as decisoes em linguagem natural**
O modulo RAG recupera contexto relevante de uma base de conhecimento (regras de negocio, playbooks de retencao) e gera uma explicacao do tipo: *"Este usuario tem alto risco principalmente pela queda de frequencia nas ultimas 3 semanas. Usuarios com esse perfil respondem melhor a descontos do que a recomendacoes de produto."*

**6. Serve predicoes em tempo quase real via API**
FastAPI com endpoints `/predict`, `/batch` e `/score`. Um evento raw de usuario chega, o feature builder constroi o vetor, o modelo pondera e retorna score + segmento + acao em milissegundos.

**7. Simula streaming de eventos**
Processador produtor/consumidor com threading que simula chegada de eventos de usuarios com delay configuravel, processando cada um em tempo quase real e calculando metricas de latencia e throughput.

**8. Visualiza tudo num dashboard executivo**
Dashboard Streamlit com KPIs globais (taxa de churn, distribuicao de scores, alto risco), segmentacao visual, analise de fairness por grupo sensivel, importancia de features e tabela de decisoes dos agentes.

**9. Rastreia experimentos com MLflow**
Cada execucao do pipeline loga metricas (ROC-AUC, precision, recall, distribuicao de scores) e parametros no MLflow, permitindo comparar rodadas e versionar modelos.

**10. Garante principios de Ethical AI**
Pseudonimiza IDs com SHA-256, remove colunas PII, converte idade em faixas etarias e analisa disparidade de churn rate por genero e pais com alertas automaticos de fairness.

---

## Arquitetura

```
Evento do usuario
      |
      v
Feature Engineering  (recency, frequency, intensity, session, trend)
      |
      v
Modelos de Churn  ->  churn_score [0..1]
      |
      +-- score < 0.4  -->  decisao direta: recommend_new_product
      |
      +-- score > 0.7  -->  decisao direta: offer_discount
      |
      +-- score 0.4-0.7  -->  Pipeline Cognitivo:
                                    |
                              AnalystAgent  (interpreta score + features)
                                    |
                              StrategyAgent  (decide acao)
                                    |
                              RAG  (gera explicacao narrativa)
                                    |
                              AuditorAgent  (valida consistencia)
                                    |
                              Langfuse  (observabilidade)
      |
      v
MLflow (metricas)  |  FastAPI (tempo real)  |  Streamlit (dashboard)
```

---

## Fases em detalhe

### Fase 1 — Engenharia de Features + Modelos de Churn

**O que faz:** transforma o dataset bruto em features que capturam comportamento real do usuario e treina tres algoritmos de classificacao.

**Engenharia de features:**
- `recency_days` — dias desde a ultima interacao. Quanto maior, maior o risco de churn
- `frequency` — frequencia de logins. Queda de frequencia e um sinal precoce de desengajamento
- `avg_session_duration` — duracao media de sessao em minutos. Sessoes mais curtas indicam perda de interesse
- `intensity` — paginas por sessao. Mede profundidade de engajamento por visita
- `engagement_trend` — placeholder para tendencia temporal (crescimento ou queda de uso)
- Codificacao de variaveis categoricas (Gender, Country, City, Payment_Method, Signup_Quarter)

**Modelos treinados:**
- `LogisticRegression` — baseline linear, rapido e interpretavel
- `RandomForestClassifier` — modelo principal, salvo em `models/rf_model.pkl` para predicao online
- `GradientBoostingClassifier` — ensemble alternativo para comparacao

**Metricas de avaliacao:** ROC-AUC, Precision, Recall

**Explicabilidade:** SHAP summary plots para entender quais features mais influenciam o modelo

---

### Fase 2 — Segmentacao Dinamica + Agentes + MLflow

**O que faz:** transforma scores de churn em segmentos acionaveis, associa acoes a cada segmento e rastreia o experimento.

**Segmentacao KMeans (n=4):**
Os clusters sao nomeados automaticamente pelo nivel de risco medio de churn:
- `engaged` — alta frequencia, sessoes longas, baixa recencia. Acao: recomendar produto novo
- `occasional` — uso moderado, engajamento variavel. Acao: campanha de engajamento
- `at_risk` — queda de frequencia, alta recencia. Acao: oferta de desconto
- `inactive` — sem interacao recente, sessoes zeradas. Acao: email de reativacao

**Segmentacao por regras (fallback):** recencia > 60 dias OU frequencia < 5 logins = at_risk

**RetentionAgent:** aplica o modelo a cada usuario do conjunto de teste, calcula `churn_score` e associa uma acao via politica de score (< 0.3 recomendacao, 0.3-0.7 sem acao, > 0.7 desconto)

**MLflow:** registra ROC-AUC, precision, recall, media/desvio de scores e contagem de alto risco por run

---

### Fase 3 — Camada Cognitiva: RAG + Agentes + Auditoria

**O que faz:** adiciona uma camada de raciocinio e explicabilidade em cima do pipeline de ML, ativada apenas para casos de incerteza.

**Filtro cognitivo (score 0.4–0.7):** a zona de incerteza onde a decisao nao e obvia. Scores extremos (<0.4 ou >0.7) recebem decisao direta sem custo computacional extra. So os casos intermediarios passam pelo pipeline completo.

**Pipeline de agentes:**

`AnalystAgent`
- Recebe: churn_score + features + contexto RAG
- Faz: interpreta o score e os sinais do usuario, classifica em risco baixo/medio/alto
- Retorna: nivel de risco, features-chave, resumo textual

`StrategyAgent`
- Recebe: resultado do analyst + acao base do modelo
- Faz: cruza nivel de risco com regras de negocio
- Logica: alto = offer_discount, medio = engagement_campaign, baixo = mantem acao base
- Retorna: acao final + raciocinio auditavel

`ChurnRAG`
- Recebe: user_id + score + features
- Faz: busca contexto relevante na base de conhecimento (regras de retencao, playbooks) usando cosine similarity com sentence-transformers, passa para o gerador
- Retorna: explicacao narrativa contextualizada
- Se OPENAI_API_KEY estiver configurada: chama GPT-4o-mini para explicacao real

`AuditorAgent`
- Recebe: estado completo (score + analysis + strategy + rag_context)
- Faz: valida consistencia da decisao contra regras de auditoria
- Exemplo de flag: score > 0.8 + acao "send_engagement_reminder" = inconsistente
- Retorna: status (ok/review) + lista de flags

**Langfuse (observabilidade):** loga cada decisao com user_id, score, acao, status de auditoria. Se chaves nao configuradas, faz fallback para log local.

---

### Fase 4 — API de Predicao Online

**O que faz:** expoe o pipeline como servico HTTP para consumo por sistemas externos (CRM, produto, plataformas de marketing).

**`src/api/app.py` — API principal (porta 8000):**
- `GET /health` — verifica se o servico esta ativo e o modelo carregado
- `POST /predict` — recebe features de um usuario, retorna score + segmento + acao + explicacao
- `POST /batch` — processa lista de usuarios em uma unica chamada

**`src/online_prediction/model_server.py` — Servidor de eventos raw (porta 8001):**
- `POST /score` — recebe evento bruto (days_since_last_purchase, login_frequency, etc.), aplica feature builder e retorna predicao

**Feature builder online:** mapeia campos brutos de eventos para o vetor de features esperado pelo modelo, com valores default para campos ausentes

**Validacao:** Pydantic v2 para validacao de input/output com tipos, ranges e exemplos documentados

---

### Fase 5 — Privacidade e Analise de Fairness

**O que faz:** garante que o pipeline respeita principios de Ethical AI e nao perpetua bias nos dados.

**Pseudonimizacao:** Customer_ID substituido por `u_` + SHA-256 truncado (16 chars). Irreversivel sem a chave de salt.

**Minimizacao de dados:** remove automaticamente colunas PII diretas (Email, Phone, Name, CPF). Relatorio mostra o antes/depois.

**Generalizacao:** coluna Age convertida em faixas etarias (<18, 18-24, 25-34, 35-44, 45-59, 60+). Impede re-identificacao por idade exata.

**Analise de fairness:** calcula taxa de churn media por grupo sensivel (Gender, Country). Alertas automaticos quando a disparidade entre grupos supera 15 pontos percentuais. Responde: *"O modelo esta tratando todos os grupos de forma equitativa?"*

---

### Fase 6 — Docker

**O que faz:** containeriza o sistema para garantir reproducibilidade e facilitar deploy.

**`docker/Dockerfile`:** imagem `python:3.10-slim` com a FastAPI. Instala dependencias, copia codigo e modelo, expoe porta 8000.

**`docker/docker-compose.yml`:** sobe dois servicos em paralelo:
- `churn-api` — FastAPI na porta 8000 com volume montado para o modelo
- `mlflow` — MLflow UI na porta 5000 apontando para o banco SQLite de experimentos

---

### Fase 7 — Kubernetes (referencia)

Templates YAML em `k8s/` para deploy em cluster:
- `deployment.yaml` — Deployment com replicas, resource limits e liveness probe
- `service.yaml` — Service para expor a API
- `kubeflow_pipeline.yaml` — Pipeline Kubeflow para treino automatizado

---

### Extras implementados

**Streaming (`src/streaming/event_processor.py`):**
Simula chegada de eventos de usuarios em tempo quase real usando arquitetura produtor/consumidor com `queue.Queue` + threading. O producer empurra eventos com delay configuravel, o consumer processa cada um (feature building + predicao + segmentacao) e retorna metricas de latencia (media, P95, max) e throughput (eventos/segundo).

**Dashboard executivo (`src/dashboard/app.py`):**
Aplicacao Streamlit com:
- KPIs: taxa de churn real, percentual de alto risco, score medio
- Histograma de scores com linhas de threshold
- Pizza de segmentacao + box plot de score por segmento
- Analise de fairness por Gender e Country com alertas
- Bar chart de importancia de features do Random Forest
- Tabela de decisoes dos agentes com distribuicao de acoes

**Experiments (`notebooks/experiments.ipynb`):**
Comparacao sistematica dos tres modelos com cross-validation 5-fold, GridSearchCV no RandomForest (n_estimators x max_depth x min_samples_split), curvas ROC sobrepostas, matriz de confusao e registro de todos os runs no MLflow.

**Benchmarks (`notebooks/benchmarks.ipynb`):**
Medicao de latencia de predicao individual (1000 runs, P95/P99), throughput batch (1 a 5000 usuarios), benchmark do pipeline cognitivo completo (analyst + strategy + RAG + auditor) e benchmark do streaming processor com 100 eventos.

**Testes (`tests/`):**
39 testes unitarios cobrindo feature engineering, modelos, clustering, agentes (analyst/strategy/auditor), politica de acoes, recomendacoes, feature builder online e segmentacao. Todos passando com pytest.

**CI/CD (`.github/workflows/ci.yml`):**
Pipeline GitHub Actions ativado em push/PR para master: lint com flake8, execucao dos 39 testes com cobertura, upload de coverage para Codecov e build da imagem Docker.

---

## Fases — Status

| Fase | Status | Descricao |
|------|--------|-----------|
| 1 | Completo | Engenharia de features + tres modelos + SHAP |
| 2 | Completo | Segmentacao KMeans + agentes + MLflow |
| 3 | Completo | Camada cognitiva: agentes + RAG + auditoria |
| 4 | Completo | FastAPI + servidor de predicao online |
| 5 | Completo | Pseudonimizacao + minimizacao + fairness |
| 6 | Completo | Docker — Dockerfile + docker-compose (API + MLflow) |
| 7 | Referencia | Kubernetes — YAML templates em `k8s/` |

### Componentes extras

| Componente | Status | Detalhes |
|------------|--------|---------|
| LLM real | Graceful | Chama GPT-4o-mini se OPENAI_API_KEY configurada, fallback para template |
| Langfuse | Graceful | Loga no Langfuse se chaves configuradas, fallback para print() |
| RAG com embeddings | Completo | SimpleRetriever usa sentence-transformers se disponivel |
| Streaming | Completo | src/streaming/event_processor.py — produtor/consumidor com threading |
| Dashboard | Completo | src/dashboard/app.py — Streamlit com KPIs, segmentacao, fairness |
| Experiments | Completo | notebooks/experiments.ipynb — cross-val, GridSearch, curvas ROC |
| Benchmarks | Completo | notebooks/benchmarks.ipynb — latencia, throughput, streaming |
| Testes | Completo | tests/ — 39 testes (features/models/agents/api) |
| CI/CD | Completo | .github/workflows/ci.yml — lint + pytest + docker build |

---

## Dataset

Dataset publico de e-commerce com 50.000 usuarios e 25 features:

- **Demograficas**: Age, Gender, Country, City, Membership_Years
- **Comportamentais**: Login_Frequency, Session_Duration_Avg, Pages_Per_Session
- **Transacionais**: Total_Purchases, Average_Order_Value, Days_Since_Last_Purchase
- **Engajamento**: Cart_Abandonment_Rate, Email_Open_Rate, Social_Media_Engagement_Score
- **Target**: `Churned` (0/1)

---

## Modelos

| Modelo | Papel |
|--------|-------|
| LogisticRegression | Baseline linear |
| RandomForestClassifier | Principal — salvo em `models/rf_model.pkl` |
| GradientBoostingClassifier | Ensemble alternativo |

Metricas: ROC-AUC, Precision, Recall. Explicabilidade via SHAP.

---

## Segmentacao

| Segmento | Perfil | Acao |
|----------|--------|------|
| `engaged` | Alta frequencia, baixa recencia | recommend_new_product |
| `occasional` | Uso moderado | engagement_campaign |
| `at_risk` | Queda de frequencia, alta recencia | offer_discount |
| `inactive` | Sem interacao recente | reengagement_email |

---

## Agentes Cognitivos

| Agente | Funcao |
|--------|--------|
| `AnalystAgent` | Interpreta score e features, define nivel de risco |
| `StrategyAgent` | Cruza analise com regras de negocio, decide acao |
| `ChurnRAG` | Recupera contexto relevante e gera explicacao narrativa |
| `AuditorAgent` | Valida consistencia da decisao e emite flags |

Filtro cognitivo: ativado apenas para scores entre 0.4 e 0.7.

---

## API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
uvicorn src.online_prediction.model_server:app --port 8001
```

| Metodo | Endpoint | Descricao |
|--------|----------|-----------|
| GET | `/health` | Status do servico |
| POST | `/predict` | Predicao individual |
| POST | `/batch` | Predicao em lote |
| POST | `/score` | Predicao por evento raw |

```json
{
  "user_id": "u_demo_001",
  "recency_days": 30.0,
  "frequency": 5.0,
  "avg_session_duration": 8.5,
  "intensity": 2.8,
  "engagement_trend": -0.3
}
```

---

## Ethical AI

| Principio | Implementacao |
|-----------|--------------|
| Pseudonimizacao | Customer_ID → SHA-256 truncado |
| Minimizacao | Remove Email, Phone, Name, CPF |
| Generalizacao | Age → faixa etaria (18-24, 25-34...) |
| Fairness | Analise de disparidade por Gender e Country com threshold de 15% |

---

## Estrutura do Projeto

```
.
├── data/
│   ├── raw/              # Dataset original (50k usuarios)
│   └── processed/        # Features engenheiradas
├── models/
│   └── rf_model.pkl      # Random Forest treinado
├── notebooks/
│   ├── main.ipynb        # Pipeline completo (fases 1-5)
│   ├── experiments.ipynb # Cross-val, GridSearch, curvas ROC
│   ├── benchmarks.ipynb  # Latencia, throughput, streaming
│   └── mlflow.db         # Tracking de experimentos
├── src/
│   ├── features/         # build_features.py
│   ├── models/           # churn_model.py, clustering_model.py
│   ├── agents/           # retention, analyst, strategy, auditor
│   ├── llm/              # rag.py, retriever.py, generator.py
│   ├── segmentation/     # user_segmentation.py
│   ├── personalization/  # actions.py
│   ├── privacy/          # anonymization.py, bias_check.py
│   ├── api/              # app.py (FastAPI), schemas.py
│   ├── online_prediction/# model_server.py, feature_builder.py
│   ├── streaming/        # event_processor.py
│   └── dashboard/        # app.py (Streamlit)
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_agents.py
│   └── test_api.py
├── .github/
│   └── workflows/ci.yml
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── k8s/                  # Templates Kubernetes
├── scripts/
└── requirements.txt
```

---

## Como Rodar

```bash
# Instalar dependencias
pip install -r requirements.txt

# Pipeline principal
jupyter notebook notebooks/main.ipynb

# Experimentos
jupyter notebook notebooks/experiments.ipynb

# Benchmarks
jupyter notebook notebooks/benchmarks.ipynb

# Dashboard
streamlit run src/dashboard/app.py

# API
uvicorn src.api.app:app --reload --port 8000

# Docker
docker compose -f docker/docker-compose.yml up --build

# Testes
pytest tests/ -v

# MLflow UI
mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db
```

### Variaveis opcionais

```bash
export OPENAI_API_KEY="sk-..."        # LLM real (GPT-4o-mini)
export LANGFUSE_PUBLIC_KEY="pk-..."   # Observabilidade
export LANGFUSE_SECRET_KEY="sk-..."
```

---

## Tecnologias

`Python 3.10` | `scikit-learn` | `pandas` | `FastAPI` | `MLflow` | `Streamlit` | `SHAP` | `sentence-transformers` | `pydantic` | `joblib` | `pytest` | `Docker`
