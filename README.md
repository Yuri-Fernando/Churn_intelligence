# Churn Intelligence Pipeline

### Customer Intelligence · Churn Prediction · AI Agents · RAG · MLflow · FastAPI

## Status

🟢 **Concluído — Projeto de portfólio / Pesquisa Aplicada em AI & Customer Intelligence**

Pipeline completo de **Churn Intelligence** desenvolvido para simular uma arquitetura moderna de **Customer Data Platform (CDP)**, transformando dados comportamentais em previsões de churn, segmentação dinâmica, decisões de retenção e insights explicáveis.

O sistema integra **Data Science, Machine Learning, agentes cognitivos, RAG, APIs, streaming, observabilidade, fairness e governança**, formando um fluxo completo entre dados brutos, predição e decisão.

---

## Sobre o Projeto

O problema central abordado é:

> **Como identificar antecipadamente que um usuário pode abandonar um serviço e determinar qual ação de retenção é mais adequada?**

O pipeline responde a essa questão em múltiplas camadas:

```text
Dados de Comportamento
        ↓
Feature Engineering
        ↓
Churn Prediction
        ↓
Segmentação
        ↓
Decision Layer
        ↓
Agentes Cognitivos
        ↓
RAG + Explicabilidade
        ↓
API / Streaming
        ↓
Dashboard + Observabilidade
        ↓
Governança / Fairness
```

---

## Objetivo

Construir uma arquitetura capaz de:

- Processar dados comportamentais;
- Criar features orientadas a comportamento;
- Estimar risco de churn;
- Explicar as predições;
- Segmentar usuários dinamicamente;
- Associar ações de retenção;
- Utilizar agentes para casos ambíguos;
- Gerar explicações contextuais com RAG;
- Validar decisões com auditoria;
- Expor predições via API;
- Processar eventos em tempo quase real;
- Rastrear experimentos;
- Avaliar fairness;
- Preparar o pipeline para evolução contínua.

---

# Pipeline Principal

```text
1. Dados Brutos
        ↓
2. Feature Engineering
        ↓
3. Modelos de Churn
        ↓
4. churn_score [0..1]
        ↓
5. Segmentação KMeans
        ↓
6. Política de Decisão
        ↓
7. Filtro Cognitivo
        ↓
8. AnalystAgent
        ↓
9. StrategyAgent
        ↓
10. ChurnRAG
        ↓
11. AuditorAgent
        ↓
12. API / Streaming
        ↓
13. MLflow / Langfuse
        ↓
14. Dashboard
```

---

# 1. Engenharia de Features + Modelos de Churn

A primeira camada transforma dados brutos em variáveis comportamentais capazes de representar dinâmica de uso.

### Features principais

| Feature | Descrição |
|---|---|
| `recency_days` | Dias desde a última interação |
| `frequency` | Frequência de logins |
| `avg_session_duration` | Duração média das sessões |
| `intensity` | Páginas por sessão |
| `engagement_trend` | Tendência de crescimento ou queda do uso |
| Variáveis categóricas | Gender, Country, City, Payment_Method, Signup_Quarter |

### Modelos

| Modelo | Papel |
|---|---|
| `LogisticRegression` | Baseline linear |
| `RandomForestClassifier` | Modelo principal para inferência online |
| `GradientBoostingClassifier` | Modelo alternativo |

O Random Forest é salvo em:

```text
models/rf_model.pkl
```

### Métricas

- ROC-AUC;
- Precision;
- Recall.

### Explainability

O pipeline utiliza **SHAP** para analisar a influência das features nas previsões.

---

# 2. Segmentação Dinâmica

Após calcular o `churn_score`, os usuários são agrupados em quatro segmentos utilizando **KMeans**.

| Segmento | Perfil | Ação |
|---|---|---|
| `engaged` | Alta frequência e baixo risco | `recommend_new_product` |
| `occasional` | Uso moderado | `engagement_campaign` |
| `at_risk` | Queda de frequência e alta recência | `offer_discount` |
| `inactive` | Sem interação recente | `reengagement_email` |

A nomenclatura dos clusters é definida a partir do nível médio de risco de churn.

### Fallback baseado em regras

Também existe uma estratégia baseada em regras para classificar usuários como `at_risk` quando:

```text
recency > 60 dias
OU
frequency < 5 logins
```

---

# 3. Política de Retenção

O `RetentionAgent` aplica o modelo de churn aos usuários e associa uma ação de retenção.

A política utiliza faixas de score:

```text
score < 0.3
    ↓
recommendation

0.3 ≤ score ≤ 0.7
    ↓
sem ação direta

score > 0.7
    ↓
offer_discount
```

Essa camada separa a **predição** da **decisão**, permitindo evoluir a política independentemente do modelo.

---

# 4. Camada Cognitiva

Os usuários que entram na **zona de incerteza** passam por uma camada adicional de análise.

```text
score < 0.4
    ↓
Decisão direta

0.4 ≤ score ≤ 0.7
    ↓
Pipeline Cognitivo

score > 0.7
    ↓
Decisão direta
```

A lógica evita utilizar agentes e RAG em situações onde a decisão já é considerada clara pelo score.

---

# Agentes Cognitivos

## AnalystAgent

Responsável por interpretar:

- `churn_score`;
- Features;
- Contexto recuperado;
- Sinais comportamentais.

Retorna:

- Nível de risco;
- Features relevantes;
- Resumo textual.

---

## StrategyAgent

Recebe a análise do AnalystAgent e a ação base do modelo.

Responsável por:

- Interpretar o nível de risco;
- Cruzar o resultado com regras de negócio;
- Definir a ação final;
- Produzir raciocínio auditável.

Exemplo:

```text
Alto risco
    ↓
offer_discount

Risco médio
    ↓
engagement_campaign

Baixo risco
    ↓
mantém ação base
```

---

## ChurnRAG

O módulo RAG recupera informações relevantes de uma base de conhecimento formada por:

- Regras de retenção;
- Playbooks;
- Contextos de negócio.

A recuperação utiliza similaridade vetorial com embeddings gerados por **sentence-transformers**.

Quando `OPENAI_API_KEY` está configurada, o sistema pode utilizar **GPT-4o-mini** para gerar a explicação narrativa.

Exemplo conceitual:

```text
Usuário apresenta queda de frequência
        ↓
RAG recupera playbook relevante
        ↓
LLM interpreta o contexto
        ↓
Explicação narrativa
```

---

## AuditorAgent

Responsável por validar a consistência da decisão.

Recebe:

- Score;
- Analysis;
- Strategy;
- Contexto RAG.

Verifica possíveis inconsistências entre o risco estimado e a ação selecionada.

Retorna:

```text
status = ok
ou
status = review
```

Também gera uma lista de flags quando identifica inconsistências.

---

# 5. API de Predição Online

O pipeline é disponibilizado através de serviços HTTP.

## API principal

```text
Porta: 8000
```

### Endpoints

| Método | Endpoint | Descrição |
|---|---|---|
| GET | `/health` | Estado do serviço |
| POST | `/predict` | Predição individual |
| POST | `/batch` | Predição em lote |

## Servidor de eventos raw

```text
Porta: 8001
```

Endpoint:

```text
POST /score
```

Recebe eventos brutos, executa o feature builder e retorna a predição.

### Feature Builder

Transforma eventos brutos em features compatíveis com o modelo.

Exemplos de entrada:

```json
{
  "days_since_last_purchase": 30,
  "login_frequency": 5,
  "session_duration": 8.5
}
```

### Validação

Utiliza **Pydantic v2** para validação dos inputs e outputs.

---

# 6. Privacidade e Ethical AI

O projeto possui uma camada específica para privacidade e análise de fairness.

## Pseudonimização

`Customer_ID` é substituído por:

```text
u_ + SHA-256 truncado
```

com 16 caracteres.

## Minimização de Dados

São removidas colunas de PII direta, incluindo:

- Email;
- Phone;
- Name;
- CPF.

## Generalização

A idade exata é transformada em faixas:

```text
<18
18-24
25-34
35-44
45-59
60+
```

## Fairness

O pipeline compara taxas de churn entre grupos sensíveis:

- Gender;
- Country.

É gerado alerta quando a disparidade ultrapassa o threshold definido de **15 pontos percentuais**.

---

# 7. MLflow

O **MLflow** é utilizado para rastrear experimentos.

Cada execução pode registrar:

- ROC-AUC;
- Precision;
- Recall;
- Média dos scores;
- Desvio dos scores;
- Quantidade de usuários de alto risco;
- Parâmetros da execução.

Isso permite comparar diferentes experimentos e manter histórico de modelos e resultados.

---

# 8. Streaming

O sistema possui um processador de eventos em:

```text
src/streaming/event_processor.py
```

O fluxo utiliza arquitetura produtor/consumidor:

```text
Producer
   ↓
queue.Queue
   ↓
Consumer
   ↓
Feature Building
   ↓
Prediction
   ↓
Segmentation
   ↓
Metrics
```

O componente simula a chegada de eventos em tempo quase real e calcula:

- Latência média;
- P95;
- Latência máxima;
- Throughput em eventos por segundo.

---

# 9. Dashboard Executivo

O dashboard utiliza **Streamlit**.

Arquivo:

```text
src/dashboard/app.py
```

### KPIs

- Taxa de churn;
- Percentual de alto risco;
- Score médio.

### Visualizações

- Histograma de scores;
- Thresholds;
- Distribuição de segmentos;
- Box plot por segmento;
- Fairness por Gender;
- Fairness por Country;
- Feature importance do Random Forest;
- Distribuição das decisões dos agentes.

---

# 10. Experimentos

O notebook:

```text
notebooks/experiments.ipynb
```

realiza comparação sistemática dos modelos.

Inclui:

- Cross-validation 5-fold;
- GridSearchCV;
- Random Forest;
- Curvas ROC;
- Matriz de confusão;
- Registro dos experimentos no MLflow.

---

# 11. Benchmarks

O notebook:

```text
notebooks/benchmarks.ipynb
```

mede:

- Latência individual;
- P95;
- P99;
- Throughput batch;
- Pipeline cognitivo completo;
- Streaming processor.

O benchmark de predição individual utiliza **1000 execuções**, enquanto o streaming é avaliado com **100 eventos**.

---

# 12. Testes

O projeto possui **39 testes unitários**, cobrindo:

- Feature engineering;
- Modelos;
- Clustering;
- AnalystAgent;
- StrategyAgent;
- AuditorAgent;
- Política de ações;
- Recomendações;
- Feature builder online;
- Segmentação;
- API.

Todos os testes estão configurados para execução via `pytest`.

---

# 13. CI/CD

O pipeline GitHub Actions é executado em:

```text
push
Pull Request
```

Etapas principais:

```text
GitHub Actions
      ↓
Flake8
      ↓
Pytest
      ↓
Coverage
      ↓
Codecov
      ↓
Docker Build
```

---

# Arquitetura Geral

```text
Evento do Usuário
        │
        ▼
Feature Engineering
        │
        ▼
Churn Model
        │
        ▼
churn_score [0..1]
        │
        ├───────────────┐
        │               │
        ▼               ▼
   Score Extremo    Score 0.4-0.7
        │               │
        │               ▼
        │        AnalystAgent
        │               │
        │               ▼
        │        StrategyAgent
        │               │
        │               ▼
        │           ChurnRAG
        │               │
        │               ▼
        │          AuditorAgent
        │               │
        └───────┬───────┘
                ▼
          Final Decision
                │
       ┌────────┼─────────┐
       ▼        ▼         ▼
    MLflow   FastAPI   Streamlit
       │
       ▼
   Langfuse
```

---

# Dataset

O projeto utiliza um dataset público de e-commerce com:

- **50.000 usuários**;
- **25 features**.

### Demográficas

- Age;
- Gender;
- Country;
- City;
- Membership_Years.

### Comportamentais

- Login_Frequency;
- Session_Duration_Avg;
- Pages_Per_Session.

### Transacionais

- Total_Purchases;
- Average_Order_Value;
- Days_Since_Last_Purchase.

### Engajamento

- Cart_Abandonment_Rate;
- Email_Open_Rate;
- Social_Media_Engagement_Score.

### Target

```text
Churned
```

---

# Modelos

| Modelo | Papel |
|---|---|
| `LogisticRegression` | Baseline linear |
| `RandomForestClassifier` | Modelo principal |
| `GradientBoostingClassifier` | Modelo alternativo |

Métricas:

- ROC-AUC;
- Precision;
- Recall.

Explainability:

- SHAP.

---

# Segmentação

| Segmento | Perfil | Ação |
|---|---|---|
| `engaged` | Alta frequência, baixa recência | `recommend_new_product` |
| `occasional` | Uso moderado | `engagement_campaign` |
| `at_risk` | Queda de frequência, alta recência | `offer_discount` |
| `inactive` | Sem interação recente | `reengagement_email` |

---

# Agentes

| Agente | Função |
|---|---|
| `AnalystAgent` | Interpreta score e features |
| `StrategyAgent` | Cruza análise com regras de negócio |
| `ChurnRAG` | Recupera contexto e gera explicação |
| `AuditorAgent` | Valida consistência da decisão |

O filtro cognitivo é aplicado somente em scores entre **0.4 e 0.7**.

---

# Estrutura do Projeto

```text
.
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── rf_model.pkl
│
├── notebooks/
│   ├── main.ipynb
│   ├── experiments.ipynb
│   ├── benchmarks.ipynb
│   └── mlflow.db
│
├── src/
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── churn_model.py
│   │   └── clustering_model.py
│   │
│   ├── agents/
│   │   ├── retention/
│   │   ├── analyst/
│   │   ├── strategy/
│   │   └── auditor/
│   │
│   ├── llm/
│   │   ├── rag.py
│   │   ├── retriever.py
│   │   └── generator.py
│   │
│   ├── segmentation/
│   │   └── user_segmentation.py
│   │
│   ├── personalization/
│   │   └── actions.py
│   │
│   ├── privacy/
│   │   ├── anonymization.py
│   │   └── bias_check.py
│   │
│   ├── api/
│   │   ├── app.py
│   │   └── schemas.py
│   │
│   ├── online_prediction/
│   │   ├── model_server.py
│   │   └── feature_builder.py
│   │
│   ├── streaming/
│   │   └── event_processor.py
│   │
│   └── dashboard/
│       └── app.py
│
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_agents.py
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── k8s/
├── scripts/
└── requirements.txt
```

---

# Como Executar

## Instalar dependências

```bash
pip install -r requirements.txt
```

## Pipeline principal

```bash
jupyter notebook notebooks/main.ipynb
```

## Experimentos

```bash
jupyter notebook notebooks/experiments.ipynb
```

## Benchmarks

```bash
jupyter notebook notebooks/benchmarks.ipynb
```

## Dashboard

```bash
streamlit run src/dashboard/app.py
```

## API

```bash
uvicorn src.api.app:app --reload --port 8000
```

## Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Testes

```bash
pytest tests/ -v
```

## MLflow

```bash
mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db
```

---

# Variáveis Opcionais

```bash
export OPENAI_API_KEY="sk-..."
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
```

Quando as credenciais do OpenAI não estão configuradas, o sistema utiliza fallback para geração baseada em template.

Quando as credenciais do Langfuse não estão configuradas, o sistema utiliza fallback para logging local.

---

# Ethical AI

| Princípio | Implementação |
|---|---|
| Pseudonimização | `Customer_ID` → SHA-256 truncado |
| Minimização | Remove Email, Phone, Name e CPF |
| Generalização | `Age` → faixa etária |
| Fairness | Análise por Gender e Country |

Threshold de alerta de disparidade:

```text
15 pontos percentuais
```

---

# Docker

A versão atual possui:

```text
docker/Dockerfile
docker/docker-compose.yml
```

O Compose executa:

```text
churn-api
    ↓
FastAPI :8000

mlflow
    ↓
MLflow UI :5000
```

---

# Kubernetes

A pasta `k8s/` contém templates de referência para execução em cluster:

```text
k8s/
├── deployment.yaml
├── service.yaml
└── kubeflow_pipeline.yaml
```

A camada Kubernetes é mantida como **referência de infraestrutura**, não como evidência de deployment produtivo.

---

# Componentes Extras

| Componente | Status |
|---|---|
| LLM real | Graceful fallback |
| Langfuse | Graceful fallback |
| RAG com embeddings | Completo |
| Streaming | Completo |
| Dashboard | Completo |
| Experiments | Completo |
| Benchmarks | Completo |
| Testes | Completo |
| CI/CD | Completo |

---

# Fases do Projeto

| Fase | Status | Descrição |
|---|---|---|
| 1 | ✅ Completo | Engenharia de features + modelos + SHAP |
| 2 | ✅ Completo | KMeans + agentes + MLflow |
| 3 | ✅ Completo | RAG + agentes + auditoria |
| 4 | ✅ Completo | FastAPI + predição online |
| 5 | ✅ Completo | Pseudonimização + minimização + fairness |
| 6 | ✅ Completo | Docker + Docker Compose |
| 7 | 🔵 Referência | Templates Kubernetes |

---

# O que este projeto demonstra

- Engenharia de dados;
- Feature Engineering;
- Machine Learning;
- Churn Prediction;
- Customer Intelligence;
- KMeans;
- Explainable AI com SHAP;
- Agentic AI;
- Multi-Agent Systems;
- RAG;
- FastAPI;
- Streaming;
- MLflow;
- Langfuse;
- Streamlit;
- Ethical AI;
- Fairness Analysis;
- Privacy by Design;
- Docker;
- Kubernetes;
- CI/CD;
- Model Serving;
- Online Prediction;
- Arquitetura modular de ML.

---

# Limitações

- O dataset utilizado é público e pode não representar todos os cenários reais de churn;
- O resultado do modelo depende das características e qualidade dos dados;
- As regras de retenção são específicas da implementação atual;
- Fairness é analisada com base nos grupos disponíveis no dataset;
- O pipeline cognitivo depende da disponibilidade dos modelos/serviços configurados;
- Kubernetes permanece como referência arquitetural;
- O sistema não constitui uma plataforma comercial completa de Customer Data Platform.

---

# Melhorias Futuras

- Monitoramento contínuo de drift;
- Model registry mais completo;
- Feature Store;
- Online learning;
- Novos modelos temporais;
- LSTM / Transformers para comportamento sequencial;
- Lead / Customer scoring avançado;
- A/B testing das ações de retenção;
- Causal inference;
- Uplift modeling;
- Integração com CRMs;
- Integração com sistemas de marketing;
- Orquestração de agentes mais avançada;
- Observabilidade distribuída;
- Deployment cloud;
- Pipeline Kubeflow completo;
- Monitoramento contínuo de fairness.

---

# Status Final

🟢 **Concluído**

O pipeline possui sua arquitetura principal implementada e validada, incluindo:

- ✅ Feature Engineering;
- ✅ Três modelos de churn;
- ✅ SHAP;
- ✅ KMeans;
- ✅ Segmentação dinâmica;
- ✅ RetentionAgent;
- ✅ AnalystAgent;
- ✅ StrategyAgent;
- ✅ ChurnRAG;
- ✅ AuditorAgent;
- ✅ FastAPI;
- ✅ Online Prediction;
- ✅ Streaming;
- ✅ Dashboard Streamlit;
- ✅ MLflow;
- ✅ Langfuse com fallback;
- ✅ Privacy layer;
- ✅ Fairness analysis;
- ✅ Docker;
- ✅ Docker Compose;
- ✅ 39 testes automatizados;
- ✅ GitHub Actions;
- ✅ Experimentos;
- ✅ Benchmarks.

A estrutura Kubernetes permanece como referência para evolução de infraestrutura.

---

# Licença

Consulte a licença definida no repositório.

---

# Autor

**Yuri Fernando Dubbern**

AI/ML Engineer · Machine Learning · Customer Intelligence · Agentic AI · Data Engineering

[LinkedIn](https://www.linkedin.com/in/yuridubbern) · [GitHub](https://github.com/Yuri-Fernando) · [Lattes](http://lattes.cnpq.br/7151392692642166) · [Linktree](https://linktr.ee/yuri.f.dubbern)
