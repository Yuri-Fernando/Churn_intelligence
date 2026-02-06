# Online Prediction - Churn

Esta pasta contém a implementação básica para **predição de churn em tempo quase real**.

## Objetivo

- Receber dados de eventos do usuário (login, compra, interação) e gerar um **score de churn** imediatamente.
- Permitir ações em tempo quase real como envio de oferta, mensagem ou notificação personalizada.
- Complementar o pipeline offline de churn com predição dinâmica.

## Como funciona

1. **Modelo offline**
   - Treinado previamente com histórico de dados.
   - Salvo em disco (`joblib` ou outro formato).
   
2. **Serviço online (API)**
   - Recebe dados do usuário via JSON.
   - Passa pelos mesmos passos de engenharia de features do modelo offline.
   - Retorna a predição do risco de churn.

3. **Pipeline simplificado**
Usuário → Evento → Feature Builder Online → Modelo Salvo → Predição → Ação

4. **Tempo quase real**
   - A predição é feita assim que o evento é recebido.
   - Ex.: login do usuário → calcula risco de churn → envia notificação em segundos.

## Como usar

1. Instale dependências:
```bash
pip install fastapi uvicorn joblib pandas scikit-learn
Salve o modelo offline:
import joblib
joblib.dump(seu_modelo_treinado, "models/churn_model.pkl")
Rode o servidor:
python src/online_prediction/model_server.py
Faça predições enviando POST requests para:
http://localhost:8000/predict
Exemplo de JSON:
{
  "user_id": 123,
  "Gender": "Female",
  "Country": "USA",
  "Login_Frequency": 5,
  "Days_Since_Last_Purchase": 10,
  "Session_Duration_Avg": 15.2,
  "Pages_Per_Session": 8
}

__________________________________________________________________________________________________________________________
1️⃣ O que significa “tempo real” aqui
Não precisa ser instantâneo milissegundos.
Geralmente “quase real” = a predição sai em segundos ou minutos depois do evento do usuário.

Exemplos:
Usuário faz login → sistema calcula risco de churn → decide enviar oferta em <1 minuto.
Usuário adiciona item no carrinho → sistema recomenda produto complementar imediatamente.


2️⃣ Como isso é feito na prática
a) Modelo treinado e salvo
Você treina seu modelo offline com todos os dados históricos (o que você fez na Fase 1).
Depois, salva o modelo usando:
joblib ou pickle (Python) para modelos sklearn.
Formatos como ONNX ou TorchScript para modelos mais complexos.

b) Endpoint ou serviço de predição
Cria uma API ou microserviço que recebe dados do usuário e retorna a predição:
Ex: Flask, FastAPI, Django, ou plataformas como Kubeflow Serving, MLflow Model Serving.

O fluxo:
Usuário realiza evento → dados vão para API.
API processa dados (mesmo pipeline de features que você já criou).
API roda o modelo → retorna risco de churn ou score de engajamento.

c) Pipeline de features “online”
No batch offline (o que você fez), você calcula features de histórico completo.
Para tempo real, algumas features precisam ser calculadas na hora:
Última compra, número de sessões nas últimas 24h, recência, frequência recente.
Isso geralmente é feito com streams de eventos:
Kafka, Kinesis, RabbitMQ ou mesmo triggers de banco de dados.

d) Ação baseada no score
Depois da predição, você decide ações:
Enviar notificação, oferta, mensagem personalizada, alerta para equipe.