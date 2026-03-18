@echo off
REM ============================================================
REM start.bat  —  Launch both backend and frontend on Windows
REM Usage: double-click or run from project root
REM ============================================================

echo.
echo =================================================
echo  CLT RAG Assistant — Local Startup (Windows)
echo =================================================

REM ── 1. Check Ollama ──────────────────────────────────────────
echo.
echo [1/4] Checking Ollama...
curl -sf http://127.0.0.1:11434/api/tags >nul 2>&1
IF ERRORLEVEL 1 (
    echo   Ollama not running. Starting server...
    start /B ollama serve
    timeout /t 3 /nobreak >nul
)
echo   Ollama ready ^✓

REM ── 2. Check dependencies ────────────────────────────────────
echo.
echo [2/4] Checking Python dependencies...
python -c "import fastapi, streamlit, ollama, faiss, qdrant_client" >nul 2>&1
IF ERRORLEVEL 1 (
    echo   Installing dependencies...
    pip install -r requirements.txt --quiet
    python -m spacy download en_core_web_sm
)
echo   Dependencies ready ^✓

REM ── 3. Check CSV ─────────────────────────────────────────────
echo.
echo [3/4] Checking data...
IF NOT EXIST "data\sample_clt_data.csv" (
    echo ERROR: CSV not found at data\sample_clt_data.csv
    echo   Copy your CLT CSV file to the data\ folder.
    pause
    exit /b 1
)
echo   CSV found ^✓

REM ── 4. Start services ────────────────────────────────────────
echo.
echo [4/4] Starting services...
echo.
echo   Backend  -^>  http://127.0.0.1:8000
echo   Frontend -^>  http://127.0.0.1:8501
echo   API Docs -^>  http://127.0.0.1:8000/docs
echo.
echo   Close this window to stop both services.
echo =================================================
echo.

REM Start backend in a new window
start "CLT RAG Backend" cmd /k "uvicorn backend.main:app --host 127.0.0.1 --port 8000"

REM Wait for backend to initialise
timeout /t 3 /nobreak >nul

REM Start frontend in a new window
start "CLT RAG Frontend" cmd /k "streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1"

echo Both services started in separate windows.
echo You can close this window.
pause
