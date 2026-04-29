#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Ensuring required directories exist..."
mkdir -p uploads
mkdir -p uploads/telegram
mkdir -p logs
