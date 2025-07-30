"""
Utility functions for visualization in sandpit refinement workflow.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


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