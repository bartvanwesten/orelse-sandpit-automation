"""
Visualization utilities for OR ELSE Sand Pit Grid Refinement Tool
"""

import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def is_codespace():
    """
    Detect if running in GitHub Codespace environment.
    
    Returns
    -------
    bool
        True if running in Codespace, False otherwise
    """
    return os.environ.get('CODESPACES') == 'true'


def get_polygon_bounds(all_polygons_list):
    """
    Calculate bounds for all polygons to set consistent axis limits.
    
    Parameters
    ----------
    all_polygons_list : list
        List of polygon sets
        
    Returns
    -------
    tuple
        (x_min, x_max, y_min, y_max)
    """
    all_polygons = []
    
    for poly_set in all_polygons_list:
        if isinstance(poly_set, list):
            if len(poly_set) > 0:
                # Check if it's a list of polygons or a single polygon
                if isinstance(poly_set[0], list) and len(poly_set[0]) > 0:
                    if isinstance(poly_set[0][0], list):
                        # List of polygons
                        all_polygons.extend(poly_set)
                    else:
                        # Single polygon
                        all_polygons.append(poly_set)
                else:
                    # Empty or invalid structure, skip
                    continue
    
    # Filter out any empty polygons
    valid_polygons = []
    for poly in all_polygons:
        if len(poly) > 0:
            try:
                poly_array = np.array(poly)
                if poly_array.shape[1] == 2:  # Valid polygon with x,y coordinates
                    valid_polygons.append(poly)
            except:
                continue
    
    if len(valid_polygons) == 0:
        return -1, 1, -1, 1  # Default bounds
    
    x_min = min(np.min(np.array(poly)[:, 0]) for poly in valid_polygons)
    x_max = max(np.max(np.array(poly)[:, 0]) for poly in valid_polygons)
    y_min = min(np.min(np.array(poly)[:, 1]) for poly in valid_polygons)
    y_max = max(np.max(np.array(poly)[:, 1]) for poly in valid_polygons)
    
    return x_min, x_max, y_min, y_max


def plot_grid(mk_object, polygons, all_refinement_polygons, all_original_polygons,
              buffer_polygons, envelope_sizes_m, n_steps, 
              original_buffer_polygons=None, title=None):
    """
    Plot meshkernel grid with sandpit polygons and refinement zones.
    
    Parameters
    ----------
    mk_object : meshkernel object
        Meshkernel object to plot
    polygons : list
        Original sandpit polygons
    all_refinement_polygons : list
        Final refinement polygons after merging
    all_original_polygons : list
        Original refinement polygons before merging
    buffer_polygons : list
        Final buffer polygons
    envelope_sizes_m : list
        Target resolutions for each step
    n_steps : int
        Number of refinement steps
    original_buffer_polygons : list, optional
        Original buffer polygons before merging (for planning phase)
    title : str, optional
        Custom plot title
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot grid using meshkernel plot_edges
    mk_object.mesh2d_get().plot_edges(ax=ax, linewidth=0.5, color='black', alpha=0.3)
    
    # Matplotlib default colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Plot original sand pit polygons with transparent fill
    for i, poly in enumerate(polygons):
        color = colors[i % len(colors)]
        poly_array = np.array(poly)
        closed_poly = np.vstack([poly_array, poly_array[0]])
        ax.plot(closed_poly[:, 0], closed_poly[:, 1], color=color, linewidth=3)
        ax.fill(closed_poly[:, 0], closed_poly[:, 1], color=color, alpha=0.4)
    
    # Plot original (non-merged) refinement step polygons first - more visible style
    for step, step_polygons in enumerate(all_original_polygons):
        for poly_idx, polygon in enumerate(step_polygons):
            color = colors[poly_idx % len(colors)]
            poly_array = np.array(polygon)
            closed_poly = np.vstack([poly_array, poly_array[0]])
            
            # Original polygons - more visible style
            ax.plot(closed_poly[:, 0], closed_poly[:, 1], 
                   color=color, linewidth=1.2, linestyle='-', alpha=0.5)
    
    # Plot merged refinement step polygons on top - bold style
    for step, step_polygons in enumerate(all_refinement_polygons):
        original_step_polygons = all_original_polygons[step]
        
        for poly_idx, polygon in enumerate(step_polygons):
            # If this step has fewer polygons than original, it means some were merged
            # Use black for merged polygons, keep original colors for non-merged
            if len(step_polygons) < len(original_step_polygons):
                # This is a merged polygon - use black
                color = 'black'
            else:
                # No merging in this step - use original sandpit color
                color = colors[poly_idx % len(colors)]
            
            poly_array = np.array(polygon)
            closed_poly = np.vstack([poly_array, poly_array[0]])
            
            # Final polygons - bold dashed style
            ax.plot(closed_poly[:, 0], closed_poly[:, 1], 
                   color=color, linewidth=2.0, linestyle='--', alpha=0.9)
    
    # Plot original buffer polygons if provided (planning phase)
    if original_buffer_polygons is not None:
        for poly_idx, polygon in enumerate(original_buffer_polygons):
            color = colors[poly_idx % len(colors)]
            poly_array = np.array(polygon)
            closed_poly = np.vstack([poly_array, poly_array[0]])
            
            ax.plot(closed_poly[:, 0], closed_poly[:, 1], 
                   color=color, linewidth=1.2, linestyle='-', alpha=0.5)
    
    # Plot final buffer polygons
    for poly_idx, polygon in enumerate(buffer_polygons):
        # If there are fewer buffer polygons than expected, some were merged
        expected_buffer_count = len(original_buffer_polygons) if original_buffer_polygons else len(polygons)
        if len(buffer_polygons) < expected_buffer_count:
            # This is a merged buffer polygon - use black
            color = 'black'
        else:
            # No merging - use original sandpit color
            color = colors[poly_idx % len(colors)]
        
        poly_array = np.array(polygon)
        closed_poly = np.vstack([poly_array, poly_array[0]])
        
        ax.plot(closed_poly[:, 0], closed_poly[:, 1], 
               color=color, linewidth=2.5, linestyle=':', alpha=0.8)
    
    ax.set_xlabel('Longitude [degrees]')
    ax.set_ylabel('Latitude [degrees]')
    ax.set_title(title if title else 'Grid with Sand Pit Polygons and Refinement Zones')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Set axis limits based on all polygons
    all_polygons_for_bounds = [polygons, all_original_polygons, all_refinement_polygons, buffer_polygons]
    if original_buffer_polygons:
        all_polygons_for_bounds.append(original_buffer_polygons)
    x_min, x_max, y_min, y_max = get_polygon_bounds(all_polygons_for_bounds)
    ax.set_xlim(x_min - 0.1, x_max + 0.1)
    ax.set_ylim(y_min - 0.1, y_max + 0.1)
    
    # Create custom legend
    legend_elements = []
    
    # Add sandpit colors as patches
    for i in range(len(polygons)):
        color = colors[i % len(colors)]
        legend_elements.append(Patch(facecolor=color, alpha=0.4, edgecolor=color, 
                                   label=f'Sand pit {i+1}'))
    
    # Add sandpit zone as black patch
    legend_elements.append(Patch(facecolor='black', alpha=0.4, edgecolor='black', 
                               label='Sandpit zone'))
    
    # Add line styles in black
    legend_elements.append(Line2D([0], [0], color='black', linewidth=2.5, linestyle=':', 
                                 label='Buffer zones'))
    legend_elements.append(Line2D([0], [0], color='black', linewidth=1.2, linestyle='-', 
                                 label='Refine (original)'))
    legend_elements.append(Line2D([0], [0], color='black', linewidth=2.0, linestyle='--', 
                                 label='Refine (merged)'))
    
    ax.legend(handles=legend_elements, loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_restart_results(datasets, polygons, title="Restart File Results"):
    """
    Plot restart file results showing partitions and bed level modifications.
    
    Parameters
    ----------
    datasets : list
        List of (partition_number, dataset) tuples
    polygons : list
        List of sandpit polygon coordinate lists
    title : str
        Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Get polygon bounds for zoomed plot
    x_min, x_max, y_min, y_max = get_polygon_bounds([polygons])
    
    # Plot 1: All partitions by color
    colors = plt.cm.tab10.colors
    
    for partition_num, dataset in datasets:
        color = colors[partition_num % len(colors)]
        
        # Plot partition points
        ax1.scatter(dataset.FlowElem_xcc.values, dataset.FlowElem_ycc.values, 
                   c=color, s=1, alpha=0.6, label=f'Partition {partition_num}')
    
    # Plot sandpit polygons on partition plot
    for i, poly in enumerate(polygons):
        poly_array = np.array(poly)
        closed_poly = np.vstack([poly_array, poly_array[0]])
        ax1.plot(closed_poly[:, 0], closed_poly[:, 1], color='red', linewidth=2)
        ax1.fill(closed_poly[:, 0], closed_poly[:, 1], color='red', alpha=0.3)
    
    ax1.set_xlabel('Longitude [degrees]')
    ax1.set_ylabel('Latitude [degrees]')
    ax1.set_title('Partitions Overview')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Zoomed view with bed levels
    all_coords = []
    all_bed_levels = []
    
    for partition_num, dataset in datasets:
        coords = np.column_stack([dataset.FlowElem_xcc.values, dataset.FlowElem_ycc.values])
        bed_levels = dataset.FlowElem_bl.values
        
        # Handle time dimension
        if len(bed_levels.shape) == 2:
            bed_levels = bed_levels[0]  # Use first time step
        
        all_coords.append(coords)
        all_bed_levels.append(bed_levels)
    
    # Combine all partitions
    combined_coords = np.vstack(all_coords)
    combined_bed_levels = np.concatenate(all_bed_levels)
    
    # Create scatter plot with bed level coloring
    scatter = ax2.scatter(combined_coords[:, 0], combined_coords[:, 1], 
                         c=combined_bed_levels, s=2, cmap='terrain_r', alpha=0.8)
    
    # Plot sandpit polygons
    for i, poly in enumerate(polygons):
        poly_array = np.array(poly)
        closed_poly = np.vstack([poly_array, poly_array[0]])
        ax2.plot(closed_poly[:, 0], closed_poly[:, 1], color='red', linewidth=3)
    
    # Set zoom to sandpit area
    margin = 0.01  # degrees
    ax2.set_xlim(x_min - margin, x_max + margin)
    ax2.set_ylim(y_min - margin, y_max + margin)
    
    ax2.set_xlabel('Longitude [degrees]')
    ax2.set_ylabel('Latitude [degrees]')
    ax2.set_title('Bed Levels (Zoomed to Sandpits)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Add colorbar for bed levels
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Bed Level [m]')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
