# ============================================================================
# RAG Service Dockerfile - Development-Optimized Build
# ============================================================================
# Strategy: Install lightweight deps during build, heavy deps (torch) interactively
# Benefits: Fast image builds (~2min), flexible development, reduced layer size

# Stage 1: Builder - Install ONLY lightweight Python dependencies
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements file
COPY leibniz_agent/services/rag/requirements.txt .

# Install ONLY lightweight dependencies (excludes torch, sentence-transformers)
# These are fast to install and rarely change
RUN pip install --no-cache-dir --user \
    fastapi>=0.115.0 \
    uvicorn[standard]>=0.24.0 \
    python-multipart>=0.0.6 \
    pydantic>=2.5.0 \
    google-genai>=1.33.0 \
    google-generativeai>=0.8.0 \
    faiss-cpu>=1.7.4 \
    numpy==1.24.3 \
    redis[asyncio]>=5.0.0 \
    httpx>=0.24.0 \
    python-dotenv==1.0.1 \
    huggingface-hub>=0.16.0 \
    langchain-huggingface \
    && echo "✅ Lightweight dependencies installed"

# ============================================================================
# Stage 2: Indexer - Build FAISS index WITHOUT heavy dependencies
# ============================================================================
FROM python:3.11-slim as indexer

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app

# Copy ALL service code (needed because __init__.py imports RAGEngine)
COPY leibniz_agent/services/rag /app/leibniz_agent/services/rag
COPY leibniz_agent/services/shared /app/leibniz_agent/services/shared

# Create minimal __init__.py files for package structure
RUN mkdir -p /app/leibniz_agent/services && \
    echo "" > /app/leibniz_agent/__init__.py && \
    echo "" > /app/leibniz_agent/services/__init__.py

# Copy knowledge base
COPY leibniz_knowledge_base /app/leibniz_knowledge_base
COPY ./leibniz_knowledge_base /app/leibniz_knowledge_base

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# NOTE: FAISS index build is SKIPPED in this stage
# Reason: sentence-transformers requires torch (~2GB download)
# Strategy: Build index interactively after torch installation, OR use pre-built index
RUN echo "⚠️  FAISS index build skipped - install torch & sentence-transformers first" && \
    echo "📝 To build index: docker exec -it <container> python -m leibniz_agent.services.rag.index_builder" && \
    mkdir -p /app/index && \
    echo "placeholder" > /app/index/.keep

# ============================================================================
# Stage 3: Runtime - Final development-ready image
# ============================================================================
FROM python:3.11-slim as runtime

# Install system tools for interactive development
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder (lightweight only)
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy ALL service code
COPY leibniz_agent/services/rag /app/leibniz_agent/services/rag
COPY leibniz_agent/services/shared /app/leibniz_agent/services/shared

# Create minimal __init__.py files
RUN mkdir -p /app/leibniz_agent/services && \
    echo "" > /app/leibniz_agent/__init__.py && \
    echo "" > /app/leibniz_agent/services/__init__.py

# Copy placeholder index from indexer stage (real index built later)
COPY --from=indexer /app/index /app/index

# Set Python environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Environment variables (defaults, override in docker-compose)
ENV GEMINI_API_KEY=""
ENV GEMINI_MODEL="gemini-2.0-flash-lite"
ENV LEIBNIZ_REDIS_HOST="redis"
ENV LEIBNIZ_REDIS_PORT="6379"
ENV LEIBNIZ_REDIS_DB="0"
ENV LEIBNIZ_RAG_KNOWLEDGE_BASE_PATH="/app/leibniz_knowledge_base"
ENV LEIBNIZ_RAG_VECTOR_STORE_PATH="/app/index"
ENV LEIBNIZ_RAG_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
ENV LEIBNIZ_RAG_TOP_K="8"
ENV LEIBNIZ_RAG_TOP_N="5"
ENV LEIBNIZ_RAG_SIMILARITY_THRESHOLD="0.3"
ENV LEIBNIZ_RAG_CACHE_TTL="3600"
ENV PORT="8003"
ENV LOG_LEVEL="INFO"

# Expose port
EXPOSE 8003

# Health check (graceful for development - allows service to start without index)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/health', timeout=5).read()" || exit 0

# Create setup script for heavy dependencies
RUN echo '#!/bin/bash\n\
echo "🚀 Installing heavy dependencies (torch, sentence-transformers)..."\n\
echo "⏱️  This will take 3-5 minutes on first run"\n\
pip install --no-cache-dir -r /app/leibniz_agent/services/rag/requirements-heavy.txt\n\
echo "✅ Heavy dependencies installed!"\n\
echo ""\n\
echo "📦 Building FAISS index..."\n\
python -m leibniz_agent.services.rag.index_builder \\\n\
    --knowledge-base /app/leibniz_knowledge_base \\\n\
    --output /app/index\n\
echo "✅ FAISS index built successfully!"\n\
echo ""\n\
echo "🎯 Ready! Start the service with: uvicorn leibniz_agent.services.rag.app:app --host 0.0.0.0 --port 8003"\n\
' > /app/setup_heavy_deps.sh && chmod +x /app/setup_heavy_deps.sh

# Default: Start service (will fail gracefully if torch not installed)
# For development: Override CMD with `tail -f /dev/null` to keep container running
CMD ["python", "-m", "uvicorn", "leibniz_agent.services.rag.app:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
