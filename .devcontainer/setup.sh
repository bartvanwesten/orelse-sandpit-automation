#!/bin/bash
set -e

echo "🚀 Setting up OR ELSE Sand Pit Environment..."

# Make sure we're using the base conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate base

# Install packages
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# Verify installations
echo "🔍 Verifying installation..."
python -c "import meshkernel; print('✅ meshkernel')"
python -c "import dfm_tools; print('✅ dfm-tools')"
python -c "import ipykernel; print('✅ ipykernel')"

echo "🎉 Setup complete!"
