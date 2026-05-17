# ── Stage 1: builder ─────────────────────────────────────────────────────────
# python:3.10-slim-bookworm already includes Python 3.10 + pip + venv.
# Build tools (gcc, git, etc.) are installed here and discarded after this stage.
FROM python:3.10-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build-only system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip and install PyTorch with CUDA support
# PyTorch CUDA wheels bundle their own libcudart/cuBLAS, so no CUDA base image needed.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy and install main requirements
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy and install module-specific requirements
# Change MODULE_REQS_CACHE_BUST value to force reinstall: --build-arg MODULE_REQS_CACHE_BUST=$(date +%s)
ARG MODULE_REQS_CACHE_BUST=1
COPY modules/ /tmp/module_reqs/
RUN find /tmp/module_reqs/ -name 'requirements.txt' -exec pip install --no-cache-dir -r {} \; || true

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
# Clean slim image — only runtime system libs + the pre-built /venv from above.
FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install runtime system dependencies and Node.js 20 in one layer
# Node.js is required at runtime by yt-dlp for JS challenge solving.
# gcc is required at runtime by triton (bitsandbytes dep) for JIT CUDA kernel compilation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    ca-certificates \
    gcc \
    libc6-dev \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy the fully built virtual environment from the builder stage
COPY --from=builder /venv /venv

ENV VENV_PATH=/venv
ENV PATH="/venv/bin:$PATH"

WORKDIR /app

# Expose the application port
EXPOSE 5001

CMD ["bash", "-c", "/venv/bin/python app.py > /app/logs/${CONTAINER_NAME:-container}_log.txt 2>&1"]