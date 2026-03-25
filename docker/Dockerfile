FROM python:3.10-slim

WORKDIR /app

# Dependencias do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar codigo fonte
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Porta da API
EXPOSE 8000

# Iniciar FastAPI
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
