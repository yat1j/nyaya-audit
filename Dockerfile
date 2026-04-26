FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc git && \
    rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch — 230 MB instead of 2.5 GB CUDA version
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu

# All other dependencies (torch is NOT in requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code only — NO model pre-download
COPY nyaya/ ./nyaya/
COPY main.py .
COPY data/ ./data/

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]