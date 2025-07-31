# OR ELSE Sand Pit Grid Refinement Tool

Automated tool for refining D-Flow FM computational grids around sand extraction areas using progressive Casulli refinement.

## Getting Started

You can use this tool in two ways:

| Feature | GitHub Codespace | Local Clone |
|---------|------------------|-------------|
| **Setup Required** | ❌ None | ✅ Python environment |
| **Interactive Polygon Drawing** | ❌ Not supported | ✅ Full support |
| **Grid Quality Analysis** | ⚠️ Slower performance | ✅ Full performance |
| **Large File Access** | ✅ Automatic | ⚠️ Manual download |
| **Cost** | ✅ Free (with limits) | ✅ Free |

### Option 1: GitHub Codespace (Recommended for Quick Start)

1. **Launch Codespace**:
   - Click the green "Code" button on GitHub → "Codespaces" → "Create Codespace"
   - Wait for environment setup (2-3 minutes)

2. **Run the Tool**:
   - Open `notebooks/automatic_sandpit_refinement.ipynb`
   - Run all cells in sequence
   - Grid file and sample polygons are pre-loaded

**Limitations**: Interactive polygon drawing not supported. Use provided sample polygons or create `.pol` files locally.

### Option 2: Local Installation

**Prerequisites**: Git, Conda/Miniconda, and optionally Git LFS

1. **Open Terminal and Navigate**:
   - **Windows**: Open Command Prompt (cmd) or PowerShell
   - **Mac/Linux**: Open Terminal
   
   Navigate to where you want to download the project:
   ```bash
   # Example: navigate to your Documents folder
   cd Documents
   # Or wherever you want to store the project
   ```

2. **Clone Repository**:
   From your chosen location, clone the repository:
   ```bash
   git clone https://github.com/bartvanwesten/orelse-sandpit-automation.git
   cd orelse-sandpit-automation
   ```

3. **Download Large Files**:
   The NetCDF grid file is for example purposes - any Delft3D Flexible Mesh grid should work. The example file is stored with Git LFS. From the project directory:
   ```bash
   # Download the example NetCDF file
   git lfs pull
   ```
   
   If you don't have Git LFS installed or the command fails:
   ```bash
   # Install Git LFS first
   git lfs install
   git lfs pull
   ```
   
   **Alternative**: If Git LFS doesn't work, you can use your own D-Flow FM NetCDF grid file - just place it in `data/input/` and update the filename in the notebook.

4. **Set Up Python Environment**:
   Still in the project directory, create and activate a conda environment:
   ```bash
   # Create conda environment
   conda create -n orelse_sandpit_env python=3.11
   
   # Activate the environment
   conda activate orelse_sandpit_env
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```

5. **Run the Tool**:
   With the environment activated:
   ```bash
   # Start Jupyter Lab
   jupyter lab
   
   # In the Jupyter interface that opens in your browser:
   # Navigate to notebooks/automatic_sandpit_refinement.ipynb and run cells
   ```
   
   **Note**: Make sure to keep the conda environment activated whenever working with the tool.

## Quick Usage

### Basic Workflow

1. **Configure** (Step 0): Set target resolution and buffer parameters
2. **Load Grid** (Step 1): Load D-Flow FM grid and sandpit polygons
3. **Plan Refinement** (Step 2): Generate refinement zones with overlap detection
4. **Execute** (Step 3): Apply Casulli refinement to grid
5. **Monitor Quality** (Step 4): Analyze grid quality metrics (optional)

### Key Parameters

```python
target_resolution = 30      # Target resolution in meters
buffer_around_sandpit = 250 # Buffer around polygons in meters  
N = 7                       # Number of transition cells
```

### Input Files

- **Grid**: NetCDF file with D-Flow FM unstructured grid (any Delft3D Flexible Mesh grid)
- **Polygons**: Either:
  - Existing `.pol` file with sandpit boundaries
  - Interactive drawing (local only)

```
├── src/                          # Core utilities
│   ├── polygon_utils.py         # Polygon operations & interactive drawing
│   ├── refinement_utils.py      # Grid analysis & Casulli refinement
│   ├── visualization_utils.py   # Plotting & environment detection
│   └── monitoring_utils.py      # Grid quality analysis
├── notebooks/
│   └── automatic_sandpit_refinement.ipynb  # Main workflow
├── data/
│   ├── input/                   # Input files (NetCDF, polygons)
│   └── output/                  # Generated outputs
└── requirements.txt             # Python dependencies
```

- `dfm_tools` - D-Flow FM utilities
- `meshkernel` - Grid operations and Casulli refinement  
- `xugrid` - Unstructured grid handling
- `shapely` - Polygon operations
- `matplotlib` - Visualization
- `numpy` - Numerical operations

## Contact

- **Author**: Bart van Westen  
- **Email**: Bart.vanWesten@deltares.nl
- **Organization**: Deltares
