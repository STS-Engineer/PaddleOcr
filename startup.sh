#!/bin/bash

# Install required system libraries for OpenCV and PaddlePaddle
apt-get update
apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev

# Set environment variables to disable OneDNN/MKLDNN (critical for Azure)
export FLAGS_use_mkldnn=False
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Start Gunicorn with explicit workers (1 worker to avoid model loading issues)
gunicorn --bind=0.0.0.0:8000 --timeout 600 --workers 1 ocr_api:app
