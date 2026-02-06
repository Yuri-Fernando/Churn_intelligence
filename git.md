#git 1
# Churn_intelligence
ML Engineering + Decision Agents + Experiment Tracking

Pipeline end-to-end para **predição de churn**, **segmentação de usuários** e **decisão inteligente de ações de retenção**, com rastreabilidade via **MLflow** e arquitetura preparada para **API, LLM, Langfuse e Dashboards**.

> Projeto em andamento — Fase 2 concluída.

Descrição:
Projeto de pesquisa aplicada em Inteligência Artificial voltado à análise de comportamento de usuários a partir de dados de eventos, com foco em engenharia de features, segmentação dinâmica, personalização em tempo quase real e predição de churn. O sistema simula um cenário de Customer Data Platform (CDP), integrando dados comportamentais, pipelines de machine learning e APIs para suporte à tomada de decisão e automação de ações de retenção, respeitando princípios de data privacy e ethical AI.

## Objetivos do Projeto

Construir um sistema de **Churn Intelligence** que vá além da predição:
- Tecnicamente sólido (ML clássico + estatística)
- Explicável (SHAP, métricas claras)
- Ético (pseudonimização, minimização de dados, bias check)
- Evolutivo (preparado para online prediction, agentes e LLMs)
- Estimar risco de churn
- Segmentar usuários
- Decidir ações personalizadas de retenção
- Garantir **observabilidade**, **fairness** e **evolução contínua**

  ## Visão Geral do Projeto

Este projeto simula um cenário real de **Customer Data Platform (CDP)** moderna, integrando:

- Dados de eventos de usuários
- Engenharia de features comportamentais
- Modelos de Machine Learning para churn
- Segmentação dinâmica de usuários
- Decisão automática de ações de retenção
- Base preparada para APIs, LLMs, tracking e dashboards

O foco não é apenas previsão, mas **tomada de decisão orientada por dados**.

## Engenharia de Features (Core do Projeto)

Features comportamentais baseadas em janelas temporais:

- **Recência**: tempo desde a última interação
- **Frequência**: número de eventos em uma janela
- **Duração média de sessão**
- **Intensidade de uso**
- **Tendência de engajamento** (↑ / ↓)
- **Tempo desde última conversão**

Essas features representam **comportamento**, não apenas atributos estáticos.
_______________________________________________________________________________________________________________________________
  ##  Arquitetura Geral

Estrutural:
Data → Feature Engineering → Modelo de Churn
↓
Retention Agent
↓
Ação de Retenção Personalizada
↓
MLflow (métricas agregadas)
Langfuse (eventos – Fase 3)
  
Conceitual: 
  Eventos de Usuário
        ↓
Pipeline de Features
        ↓
Modelagem Comportamental
        ↓
Predição de Churn
        ↓
API de Personalização / Retenção

______________________________________________________________________________________________________________________
##  Fase 2 — Status Atual

### Implementado
- Modelo de churn (RandomForest)
- Agente de decisão (`RetentionAgent`)
- Segmentação automática:
  - `engaged`
  - `neutral`
  - `at_risk`
- Tracking com **MLflow**
  - métricas agregadas
  - parâmetros de execução
- Análise de **bias / fairness**
- Pipeline rodando em batch (50k usuários)

### Exemplo de Métricas (MLflow)
- churn_score_mean
- churn_score_std
- churn_score_min / max
- parâmetros de execução e fase

---

##  Decisões Inteligentes

O sistema **não apenas prevê churn** — ele decide ações:

- `offer_standard_discount`
- `offer_premium_discount`
- `recommend_new_product`
- ações personalizadas por perfil

Separação clara entre:
- **modelo**
- **regra**
- **decisão**

---

##  Fairness & Bias

Análise automática de bias por atributos sensíveis (ex: gênero, país), garantindo:
- visibilidade de possíveis desvios
- base para mitigação futura

---

##  Roadmap
O que já está implementado / funcional:
- Pipeline de features comportamentais: recência, frequência, intensidade, tendência de engajamento, codificação de variáveis categóricas.
- Modelos de predição de churn: LogisticRegression, RandomForest, GradientBoosting com avaliação de métricas (roc_auc, precision, recall).
- Segmentação dinâmica de usuários: Engaged, Neutral, At Risk (com thresholds ajustáveis).
- Ações recomendadas por segmento: recomendar produtos, ofertas de desconto, ou nenhuma ação.
-Refino da segmentação: thresholds melhores, neutral segment e granularidade mais fina.
-Ações inteligentes: ofertas personalizadas, lembretes, recomendações baseadas no score do churn e no comportamento do usuário.
-Integração com LLMs e RAG: gerar insights e conteúdos dinâmicos para retenção ou personalização.
-Agentes inteligentes: agentes de retenção que decidem ações automaticamente.
-Tracking de decisões: integração com Langfuse para rastrear decisões de LLMs e agentes.
-Tracking de experimentos: integração com MLflow para monitorar métricas e resultados de modelos.
-Análises de privacidade e bias: módulo src/privacy/ para análises éticas e fairness de decisões de ML/AI.

### Futuro:
- LLM/RAG/integração API (FastAPI) (não local)
- Containerização e orquestração: Docker + Kubernetes (para produção real).
- Experimentos detalhados e benchmarks: pasta experiments/ para testes de performance e comparação de modelos.
- Streaming / tempo real
- Kubernetes (opcional)
- Langfuse
- 
### Fase 3
- FastAPI (online prediction)
- Langfuse (eventos por usuário)
- Integração LLM (RAG / explicabilidade)
- Dash executivo

###  Fase 4
- Docker
- Kubernetes
- CI/CD
- Monitoramento em produção

---
## 🧑‍💻 Autor

Projeto desenvolvido por **Yuri**  
Foco em:
- ML Engineering
- Sistemas inteligentes
- Produto de dados
- Observabilidade e decisão automatizada

---

