#!/usr/bin/env bash
# ============================================================
# Render Deployment — Build Script
# Installs system dependencies + Python packages
# ============================================================
set -o errexit

# System deps for PyMuPDF
apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev mupdf-tools curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
