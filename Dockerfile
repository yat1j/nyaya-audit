FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc git && \
    rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch — 230 MB not 2.5 GB
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY nyaya/ ./nyaya/
COPY main.py .
COPY data/ ./data/

EXPOSE 8080

# Uses Railway's PORT variable — fixes "Application failed to respond"
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]