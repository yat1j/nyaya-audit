FROM python:3.11-slim

WORKDIR /app

# Only gcc needed — no git, no torch, no sentence-transformers
RUN apt-get update && apt-get install -y gcc && \
    rm -rf /var/lib/apt/lists/*

# No torch install — model runs via HuggingFace Inference API
# Image size: ~400 MB total (well under Railway 4 GB limit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY nyaya/ ./nyaya/
COPY main.py .
COPY data/ ./data/

EXPOSE 8080

# ${PORT:-8080} uses Railway's injected PORT variable
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]