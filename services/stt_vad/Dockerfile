# ============================================================================
# STT/VAD Microservice - Docker Image
# ============================================================================
# Multi-stage build for optimized production image
#
# Reference: leibniz_agent/docs/Cloud Transformation.md - Phase 2
# ============================================================================

# Stage 1: Builder
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libsndfile1-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
WORKDIR /build
COPY leibniz_agent/services/stt_vad/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy application code (microservice only, not full leibniz_agent)
COPY leibniz_agent/services/stt_vad /app/leibniz_agent/services/stt_vad
COPY leibniz_agent/services/shared /app/leibniz_agent/services/shared

# Create minimal __init__.py files (no dependencies on main agent)
RUN echo "" > /app/leibniz_agent/__init__.py && \
    echo "" > /app/leibniz_agent/services/__init__.py

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Expose service port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/health', timeout=5)"

# Run application (CRITICAL: --workers 1 to avoid singleton race conditions)
CMD ["python", "-m", "uvicorn", "leibniz_agent.services.stt_vad.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1"]
