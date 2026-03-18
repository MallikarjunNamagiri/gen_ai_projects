#!/bin/bash
# ============================================================
# start.sh  —  Launch both backend and frontend together
# Usage:  chmod +x start.sh && ./start.sh
# ============================================================

set -e

echo ""
echo "================================================="
echo " CLT RAG Assistant — Local Startup"
echo "================================================="

# ── 1. Check Ollama ──────────────────────────────────────────
echo ""
echo "[1/4] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "ERROR: ollama not found."
    echo "  Install: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

if ! curl -sf http://127.0.0.1:11434/api/tags > /dev/null; then
    echo "  Ollama not running — starting server in background..."
    ollama serve &
    sleep 3
fi

MODEL=${OLLAMA_MODEL:-llama3.2}
if ! ollama list | grep -q "$MODEL"; then
    echo "  Pulling model '$MODEL' (first time only)..."
    ollama pull "$MODEL"
fi
echo "  Ollama ready with model '$MODEL' ✓"

# ── 2. Check virtual env / dependencies ──────────────────────
echo ""
echo "[2/4] Checking Python dependencies..."
if ! python -c "import fastapi, streamlit, ollama, faiss, qdrant_client" 2>/dev/null; then
    echo "  Installing dependencies..."
    pip install -r requirements.txt --quiet
    python -m spacy download en_core_web_sm --quiet
fi
echo "  Dependencies ready ✓"

# ── 3. Check CSV ──────────────────────────────────────────────
echo ""
echo "[3/4] Checking data..."
CSV=${CSV_PATH:-./data/sample_clt_data.csv}
if [ ! -f "$CSV" ]; then
    echo "ERROR: CSV not found at '$CSV'"
    echo "  Copy your CLT CSV to: $CSV"
    echo "  Or set: export CSV_PATH=/path/to/your/file.csv"
    exit 1
fi
echo "  CSV found: $CSV ✓"

# ── 4. Start backend and frontend ────────────────────────────
echo ""
echo "[4/4] Starting services..."
echo ""
echo "  Backend  →  http://127.0.0.1:8000"
echo "  Frontend →  http://127.0.0.1:8501"
echo "  API Docs →  http://127.0.0.1:8000/docs"
echo ""
echo "  Press Ctrl+C to stop both services."
echo "================================================="
echo ""

# Run backend in background, frontend in foreground
uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Give backend a moment to start
sleep 2

# Trap Ctrl+C to kill both
trap "echo ''; echo 'Stopping...'; kill $BACKEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1
