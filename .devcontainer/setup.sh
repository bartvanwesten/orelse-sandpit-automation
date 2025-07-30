#!/bin/bash
set -e

echo "🚀 Setting up OR ELSE Sand Pit Environment..."

# Pull LFS files if they exist
echo "📥 Downloading LFS files..."
git lfs pull || echo "No LFS files to download"

# Install packages
echo "📦 Installing Python packages..."
pip install -r requirements.txt

echo ""
echo "🎉 Environment setup complete!"
echo "📓 Open 'notebooks/automatic_sandpit_refinement.ipynb' to get started"
echo "🐍 Python interpreter: /opt/conda/bin/python"
