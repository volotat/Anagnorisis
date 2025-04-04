# Use NVIDIA CUDA base image with Python support
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment and install Python dependencies
RUN python3 -m venv .env
# Add venv to PATH
ENV PATH="/app/.env/bin:$PATH"

# Upgrade pip and install PyTorch with CUDA support
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy requirements file
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 5001

# Default command to run the application using the script
COPY run.sh /app/
RUN chmod +x /app/run.sh

# Set the default command to run the script with proper working directory
CMD ["bash", "-c", "/app/run.sh > /app/container_log.txt 2>&1"]