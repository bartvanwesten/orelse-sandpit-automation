"""
Orelse Sand Pit Grid Refinement Automation

Automated tool for refining D-Flow FM computational grids around sand extraction areas.
Uses HYDROLIB-core and dfm-tools to apply progressive Casulli refinement.
"""

__version__ = "0.1.0"
__author__ = "Bart van Westen"
__email__ = "Bart.vanWesten@deltares.nl"  # Update with your actual email

# Import main utility modules to make them easily accessible
from .polygon_utils import (
    InteractivePolygonDrawer,
    load_pol_file,
    save_pol_file,
    generate_refinement_polygons
)

from .refinement_utils import (
    compute_refinement_steps,
    apply_casulli_refinement,
    print_refinement_summary
)

from .visualization_utils import plot_grid

from .monitoring_utils import (
    analyze_grid_quality,
    plot_grid_quality
)

# Define what gets imported with "from src import *"
__all__ = [
    "InteractivePolygonDrawer",
    "load_pol_file", 
    "save_pol_file",
    "generate_refinement_polygons",
    "compute_refinement_steps",
    "apply_casulli_refinement",
    "print_refinement_summary",
    "plot_grid",
    "analyze_grid_quality",
    "plot_grid_quality"
]
