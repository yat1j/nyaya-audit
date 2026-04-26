FROM python:3.11-slim

WORKDIR /app

# Install system build tools
RUN apt-get update && apt-get install -y gcc git && \
    rm -rf /var/lib/apt/lists/*

# ── STEP 1: Install PyTorch CPU-only FIRST ─────────────────────────────────
# This is the critical fix.
# Default: pip installs torch with CUDA = 2.5 GB → exceeds Railway 4 GB limit
# CPU-only: torch = 230 MB → total image stays under 1.8 GB
# The --index-url flag tells pip to get torch from PyTorch's own server
# which hosts CPU-only wheels. Your code works identically — no API changes.
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ── STEP 2: Install all other dependencies ──────────────────────────────────
# torch is NOT in requirements.txt (we already installed it above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── STEP 3: Pre-download e5-large model during build ────────────────────────
# This eliminates cold start during the demo.
# With CPU-only PyTorch (230 MB) + model (2.1 GB) + everything else (1.1 GB)
# the total image is ~1.8 GB — well under Railway's 4 GB limit.
# Without this line the model downloads on first API call (takes ~3 minutes).
# Keep this line to avoid that delay during your demo.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
print('Downloading multilingual-e5-large...'); \
SentenceTransformer('intfloat/multilingual-e5-large'); \
print('Model downloaded and cached.')"

# ── STEP 4: Copy application code ───────────────────────────────────────────
COPY nyaya/ ./nyaya/
COPY main.py .
COPY data/ ./data/

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]