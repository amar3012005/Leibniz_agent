# Stage 1: Builder
FROM python:3.11-slim AS builder
WORKDIR /build
COPY leibniz_agent/services/stt_vad/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

# Install OS packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy artifacts and code
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Configure env vars and port
ENV GEMINI_API_KEY=""
ENV LEIBNIZ_REDIS_HOST="redis"
ENV LEIBNIZ_REDIS_PORT="6379"
ENV LEIBNIZ_REDIS_DB="0"
EXPOSE 8001

# Add HEALTHCHECK
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/health', timeout=5.0).raise_for_status()"

# Entrypoint
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
