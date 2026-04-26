FROM python:3.11-slim

WORKDIR /app

# System build tools — gcc needed by some pip packages
RUN apt-get update && apt-get install -y gcc git && \
    rm -rf /var/lib/apt/lists/*

# ── CRITICAL: Install CPU-only PyTorch FIRST ──────────────────────────────────
# Must be done before requirements.txt because pip would otherwise
# install the CUDA version (2.5 GB) when it sees torch in requirements.
# CPU-only wheel = 230 MB. CUDA wheel = 2.5 GB. Railway limit = 4 GB.
# The --index-url flag points pip to PyTorch's CPU-only package server.
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ── Install all other dependencies ────────────────────────────────────────────
# torch is NOT in requirements.txt — it is already installed above.
# If torch appears in requirements.txt it will reinstall the CUDA version.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
# No model pre-download here — e5-small downloads at first API call (~30 sec).
# Pre-downloading e5-small adds 70 MB (acceptable) but pre-downloading
# e5-large would add 2.1 GB and push image back over 4 GB.
# For the demo: hit /health once before presenting — wakes the container
# and triggers the 30-second e5-small download. Then all calls are instant.
COPY nyaya/ ./nyaya/
COPY main.py .
COPY data/ ./data/

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]