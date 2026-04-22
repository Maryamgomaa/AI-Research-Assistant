#!/bin/bash
# ============================================================
# Jetson Nano Setup Script
# AI Research Assistant — Full Stack
# Run as: bash setup_jetson.sh
# ============================================================

set -euo pipefail

echo "======================================================"
echo "  AI Research Assistant — Jetson Nano Setup"
echo "======================================================"

# ── 1. System dependencies ──────────────────────────────────
echo "[1/7] Installing system dependencies..."
sudo apt-get update -q
sudo apt-get install -y \
  python3-pip python3-venv git curl \
  libmupdf-dev mupdf-tools \
  libopenblas-dev liblapack-dev \
  libjpeg-dev libpng-dev

# ── 2. Python virtual environment ──────────────────────────
echo "[2/7] Creating Python venv..."
python3 -m venv /opt/research_assistant_venv
source /opt/research_assistant_venv/bin/activate

# ── 3. Python packages ─────────────────────────────────────
echo "[3/7] Installing Python packages..."
pip install --upgrade pip wheel setuptools

# Jetson Nano: PyTorch must be the ARM/CUDA wheel from NVIDIA
# If not already installed, get the JetPack-compatible wheel:
# pip install torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v511/

pip install \
  fastapi uvicorn httpx \
  sentence-transformers \
  PyMuPDF \
  qdrant-client \
  pydantic \
  python-multipart \
  python-dotenv \
  numpy

echo "[3/7] ✓ Python packages installed"

# ── 4. Qdrant via Docker ───────────────────────────────────
echo "[4/7] Starting Qdrant vector database..."
if ! command -v docker &> /dev/null; then
  echo "  Docker not found — installing..."
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker $USER
fi

docker pull qdrant/qdrant:latest
docker stop qdrant 2>/dev/null || true
docker rm qdrant 2>/dev/null || true
docker run -d \
  --name qdrant \
  --restart unless-stopped \
  -p 6333:6333 \
  -p 6334:6334 \
  -v /opt/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

echo "[4/7] ✓ Qdrant running on port 6333"

# ── 5. Ollama ──────────────────────────────────────────────
echo "[5/7] Installing Ollama..."
if ! command -v ollama &> /dev/null; then
  curl -fsSL https://ollama.ai/install.sh | sh
fi

# Start Ollama service
ollama serve &
sleep 3

# Pull recommended models
echo "  Pulling mistral (general reasoning)..."
ollama pull mistral

echo "  Pulling aya (multilingual, good Arabic support)..."
ollama pull aya  # Cohere's multilingual model — strong Arabic

echo "[5/7] ✓ Ollama running with mistral + aya"

# ── 6. Download embedding model ─────────────────────────────
echo "[6/7] Pre-downloading embedding model..."
python3 -c "
from sentence_transformers import SentenceTransformer
print('  Downloading paraphrase-multilingual-mpnet-base-v2...')
m = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
print(f'  ✓ Model ready. Vector dim = {m.get_sentence_embedding_dimension()}')
"

# ── 7. Create Qdrant collection ─────────────────────────────
echo "[7/7] Initialising Qdrant collection..."
python3 -c "
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff
client = QdrantClient('localhost', port=6333)
existing = {c.name for c in client.get_collections().collections}
if 'research_papers' not in existing:
    client.create_collection(
        collection_name='research_papers',
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100)
    )
    print('  ✓ Collection research_papers created')
else:
    print('  ✓ Collection already exists')
"

echo ""
echo "======================================================"
echo "  ✅ SETUP COMPLETE"
echo "======================================================"
echo ""
echo "  Qdrant:   http://localhost:6333"
echo "  Ollama:   http://localhost:11434"
echo "  Worker:   start with: uvicorn worker.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "  To start the worker:"
echo "    source /opt/research_assistant_venv/bin/activate"
echo "    cd /path/to/ai_research_assistant"
echo "    uvicorn worker.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
