#!/bin/bash
set -e

echo "🚀 Setting up OR ELSE Sand Pit Environment..."

# Pull LFS files if they exist
echo "📥 Downloading LFS files..."
git lfs pull || echo "No LFS files to download"

# Install packages
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# Set up Python kernel for Jupyter
echo "🔧 Configuring Jupyter kernel..."
python -m ipykernel install --user --name=python3 --display-name="Python 3 (ipykernel)"

# Set up Jupyter extensions (skip if fails)
echo "🔧 Setting up Jupyter extensions..."
jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build 2>/dev/null || echo "Extension setup skipped"

echo ""
echo "🎉 Environment setup complete!"
echo "📓 Open 'notebooks/automatic_sandpit_refinement.ipynb' to get started"
echo "🐍 Python interpreter: /opt/conda/bin/python"
