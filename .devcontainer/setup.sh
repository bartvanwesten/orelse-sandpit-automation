#!/bin/bash
set -e  # Exit on any error

echo "🚀 Setting up OR ELSE Sand Pit Environment..."

# System libraries are already installed via Dockerfile
echo "📦 Installing Python packages..."
pip install --no-cache-dir -r requirements.txt

# Verify key packages are installed
echo "🔍 Verifying installation..."
python -c "import meshkernel; print('✅ meshkernel installed')"
python -c "import dfm_tools; print('✅ dfm-tools installed')"
python -c "import xarray; print('✅ xarray installed')"
python -c "import netCDF4; print('✅ netCDF4 installed')"

# Test NetCDF file reading capability
echo "🧪 Testing NetCDF capabilities..."
python -c "import xarray as xr; print('✅ xarray NetCDF engines:', xr.backends.list_engines())"

echo ""
echo "🎉 Environment setup complete!"
echo "📓 Open 'notebooks/automatic_sandpit_refinement.ipynb' to get started"
echo "🐍 Python interpreter: /opt/conda/bin/python"
