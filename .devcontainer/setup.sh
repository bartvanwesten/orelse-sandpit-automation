#!/bin/bash
set -e  # Exit on any error

echo "🚀 Setting up OR ELSE Sand Pit Environment..."

# Use base conda environment (more reliable in Codespaces)
echo "📦 Installing Python packages..."
pip install --no-cache-dir -r requirements.txt

# Verify key packages are installed
echo "🔍 Verifying installation..."
python -c "import meshkernel; print('✅ meshkernel installed')"
python -c "import dfm_tools; print('✅ dfm-tools installed')"
python -c "import xarray; print('✅ xarray installed')"
python -c "import matplotlib; print('✅ matplotlib installed')"

# Set up Jupyter extensions
echo "🔧 Setting up Jupyter extensions..."
jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build 2>/dev/null || echo "Extension already installed"

echo ""
echo "🎉 Environment setup complete!"
echo "📓 Open 'notebooks/automatic_sandpit_refinement.ipynb' to get started"
echo "🐍 Python interpreter: /opt/conda/bin/python"
