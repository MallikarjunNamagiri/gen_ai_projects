from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.api.endpoints import router as api_router
from src.config import DATA_DIR

app = FastAPI(title="Document AI API", version="2.0")

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prefix all API endpoints
app.include_router(api_router, prefix="/api")

# Serve uploaded documents statically so the frontend can open them directly
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(DATA_DIR)), name="files")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Document AI Backend"}