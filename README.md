# OR ELSE Sand Pit Grid Refinement Tool

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/yourusername/your-repo-name)

Grid refinement tool for D-Flow FM computational meshes around sand extraction areas in the North Sea. Uses Meshkernel to apply Casulli refinement within user-defined polygons representing sand pits.

## About OR ELSE

This tool is part of the [OR ELSE research project](https://or-else.nl/en/) - a 5-year interdisciplinary research programme studying the ecological effects of sand extraction on the North Sea floor. The project aims to gather knowledge for ecologically responsible sand extraction as demand grows due to sea level rise and coastal protection needs.

## Features

- **Interactive polygon drawing** or loading from existing .pol files
- **Progressive grid refinement** using Casulli refinement method
- **Multiple refinement steps** with smooth transitions between resolution levels
- **Overlap detection and merging** of refinement zones
- **Grid quality analysis** and visualization tools
- **Target resolution** down to 30 meters

## How to Use

### GitHub Codespaces (Recommended)

1. **Click the "Open in GitHub Codespaces" button above**
2. **Wait 3-5 minutes** for automatic environment setup
3. **Open**: `notebooks/automatic_sandpit_refinement.ipynb`
4. **Follow the notebook workflow** to refine your grid
5. **Download results** from `data/output/`

### Local Installation

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
conda env create -f environment.yml
conda activate orelse-env
jupyter lab notebooks/automatic_sandpit_refinement.ipynb
```

## Workflow

1. **Configure** refinement parameters (target resolution, buffer distance)
2. **Load** D-Flow FM grid and define/load sand pit polygons  
3. **Generate** refinement zones with overlap detection
4. **Apply** Casulli refinement to the mesh
5. **Analyze** grid quality (optional)

## Requirements

- Python 3.11+
- Meshkernel
- dfm-tools  
- numpy, matplotlib, shapely
- Jupyter

## Input/Output

**Input:** D-Flow FM network files (.nc), sand pit polygons (.pol)  
**Output:** Refined mesh files, quality analysis, visualization plots

## Contact

**Bart van Westen**  
Email: Bart.vanWesten@deltares.nl  
Deltares

## License

MIT License
