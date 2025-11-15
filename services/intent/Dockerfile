# Multi-stage build for Intent Classification Service
# Phase 3 - Cloud Transformation

# ============================================================================
# Builder Stage
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies (none needed for pure Python service)
# All dependencies are pure Python (no system libs required unlike STT/VAD)

# Copy requirements and install dependencies
COPY leibniz_agent/services/intent/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ============================================================================
# Runtime Stage
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Update PATH to use installed packages
ENV PATH=/root/.local/bin:$PATH

# Copy service code
COPY leibniz_agent/services/intent/__init__.py ./
COPY leibniz_agent/services/intent/config.py ./
COPY leibniz_agent/services/intent/patterns.py ./
COPY leibniz_agent/services/intent/intent_classifier.py ./
COPY leibniz_agent/services/intent/app.py ./

# Expose service port
EXPOSE 8002

# Health check (using stdlib urllib - no external dependencies needed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health', timeout=5)" || exit 1

# Run with 1 worker to maintain consistent metrics and avoid state fragmentation
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
