FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

ENV LLM_PROVIDER=ollama
ENV OLLAMA_BASE_URL=http://ollama:11434
ENV OLLAMA_MODEL=llama3.1
ENV PGVECTOR_URL=postgresql://recouser:recopass@pgvector:5432/recommendations
ENV ELASTICSEARCH_URL=http://elasticsearch:9200
ENV API_PORT=8000

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
