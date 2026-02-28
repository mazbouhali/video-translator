# Arabic Video Translator
# Multi-stage build for smaller final image

# ==============================================================================
# Stage 1: Build environment with CUDA support
# ==============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# ==============================================================================
# Stage 2: Install Python dependencies
# ==============================================================================
FROM base AS deps

COPY requirements.txt .

# Install PyTorch with CUDA + other deps
RUN pip install --upgrade pip && \
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# ==============================================================================
# Stage 3: Pre-download models (optional but recommended)
# ==============================================================================
FROM deps AS models

# Download Whisper model during build (faster startup)
# Comment out to download on first run instead
RUN python -c "import whisper; whisper.load_model('medium')" && \
    python -c "import whisper; whisper.load_model('small')"

# ==============================================================================
# Stage 4: Final runtime image
# ==============================================================================
FROM models AS runtime

# Copy application
COPY main.py .
COPY app/ ./app/

# Create mount points
RUN mkdir -p /app/input /app/output /app/temp /app/models

# Model cache locations
ENV TRANSFORMERS_CACHE=/app/models \
    HF_HOME=/app/models \
    TORCH_HOME=/app/models \
    XDG_CACHE_HOME=/app/models

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import whisper; print('OK')" || exit 1

# Default: watch mode
CMD ["python", "main.py", "--watch"]
