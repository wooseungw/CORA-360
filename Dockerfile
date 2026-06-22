# PanoLLaVA Dockerfile
# ===================

# Use Python 3.11 slim image as base
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd --create-home --shell /bin/bash panollava
USER panollava

# Set work directory
WORKDIR /home/panollava

# Copy requirements first for better caching
COPY --chown=panollava:panollava tools/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --user --upgrade pip && \
    pip install --user -r /tmp/requirements.txt

# Add user bin to PATH
ENV PATH="/home/panollava/.local/bin:${PATH}"

# Copy project files
COPY --chown=panollava:panollava . /home/panollava/panollava/

# Install PanoLLaVA in development mode
RUN cd /home/panollava/panollava && pip install --user -e .

# Set the working directory to the project
WORKDIR /home/panollava/panollava

# Default command
CMD ["python", "scripts/simple_inference.py", "--help"]

# =============================================================================
# Development stage
# =============================================================================
FROM base as development

# Install development dependencies
RUN pip install --user -e ".[dev]"

# Keep container running for development
CMD ["tail", "-f", "/dev/null"]

# =============================================================================
# Production stage
# =============================================================================
FROM base as production

# Create volume for persistent data
VOLUME ["/home/panollava/panollava/data", "/home/panollava/panollava/results"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import panovlm; print('PanoLLaVA is healthy')" || exit 1

# Default command for production
CMD ["python", "scripts/simple_inference.py"]

# =============================================================================
# GPU-enabled stage (requires NVIDIA Container Toolkit)
# =============================================================================
FROM python:3.11-slim as gpu

# Install CUDA runtime (adjust version as needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Create user and setup environment
RUN useradd --create-home --shell /bin/bash panollava
USER panollava
WORKDIR /home/panollava

# Copy and install dependencies
COPY --chown=panollava:panollava tools/requirements.txt /tmp/requirements.txt
RUN pip install --user --upgrade pip && \
    pip install --user -r /tmp/requirements.txt

ENV PATH="/home/panollava/.local/bin:${PATH}"

# Copy project
COPY --chown=panollava:panollava . /home/panollava/panollava/
RUN cd /home/panollava/panollava && pip install --user -e .

WORKDIR /home/panollava/panollava

# GPU-specific default command
CMD ["python", "-c", "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"]