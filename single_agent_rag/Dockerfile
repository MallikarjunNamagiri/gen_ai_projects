# Use a lightweight official Python runtime
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8501

# Install essential system dependencies (build-essential for C-extensions, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install CPU-optimized PyTorch first to reduce image size
COPY requirements.txt .

# Install CPU wheels for PyTorch first, then install the remaining requirements
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt


# Install CPU-specific wheels for both torch and torchvision
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining project requirements
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models during build time so the container starts instantly without internet dependency
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy project source code
COPY src/ ./src/
COPY app.py .

# Create persistent storage directories
RUN mkdir -p /app/data /app/faiss_index /app/faiss_query_cache

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to ensure container is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit with container-friendly flags
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--server.enableCORS=false", \
            "--server.enableXsrfProtection=false"]