#!/bin/bash
set -e

echo "🚀 Setting up OR ELSE Sand Pit Environment..."

# Pull LFS files if they exist
echo "📥 Downloading LFS files..."
git lfs pull || echo "No LFS files to download"

# Install packages
echo "📦 Installing Python packages..."
pip install -r requirements.txt

echo "🎉 Setup complete!"
