FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_CACHE=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache

WORKDIR /app

# Toolchain for building tokenizers/onnx wheels + curl for healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements-docker.txt

COPY app ./app

# Bake embedding + reranker weights into the image so containers boot fast
RUN python -c "from app.core.vectorstore import embeddings; from app.services.reranker import rerank; print('models cached')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
