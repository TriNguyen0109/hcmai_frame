# PyTorch CUDA 12.1 Base Image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set non-interactive environment for apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies (FFmpeg, OpenCV dependencies, Git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies (including tensorflow, torch, transformers, faiss, opencv)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and TransNetV2 repo
COPY TransNetV2 /app/TransNetV2
COPY src /app/src
COPY pipeline.py /app/pipeline.py
COPY search.py /app/search.py

# Install TransNetV2 package into container python environment
RUN pip install --no-cache-dir -e /app/TransNetV2

# Create directories for data and storage
RUN mkdir -p /app/data /app/storage

# Default Command
CMD ["python", "pipeline.py", "--help"]
