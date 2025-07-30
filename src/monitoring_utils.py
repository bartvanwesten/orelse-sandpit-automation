"""
Utility functions for grid quality monitoring in sandpit refinement workflow.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as colors
import dfm_tools as dfmt


def analyze_grid_quality(mk_object, ugrid_original, all_refinement_polygons, polygons):
    """
    Analyze grid quality metrics after refinement.
    
    Parameters
    ----------
    mk_object : meshkernel object
        Refined meshkernel object
    ugrid_original : xugrid dataset
        Original grid dataset
    all_refinement_polygons : list
        Refinement polygons
    polygons : list
        Original sandpit polygons
        
    Returns
    -------
    dict
        Dictionary containing quality metrics and analysis results
    """
    # Define buffer polygon for analysis (first refinement step)
    buffer_polygon = all_refinement_polygons[0][0]
    buffer_coords = np.array(buffer_polygon)
    buffer_path = mpath.Path(buffer_coords)
    
    # Convert meshkernel grid to ugrid for face area calculation
    ugrid_refined = dfmt.meshkernel_to_UgridDataset(mk_object, crs='EPSG:4326')
    
    # Get face coordinates and areas
    face_coords = ugrid_refined.grid.face_coordinates
    face_areas = ugrid_refined.grid.area
    
    # Filter faces within buffer polygon
    faces_in_buffer = buffer_path.contains_points(face_coords)
    faces_indices_in_buffer = np.where(faces_in_buffer)[0]
    
    print("Faces in buffer area:", len(faces_indices_in_buffer), "/", len(face_coords))
    
    # Calculate resolution for all faces (sqrt of face areas)
    mean_lat = np.mean(face_coords[:, 1])
    earth_radius = 6371000
    lat_rad = np.radians(mean_lat)
    deg_to_m_lon = earth_radius * np.cos(lat_rad) * np.pi / 180
    deg_to_m_lat = earth_radius * np.pi / 180
    deg_to_m2 = deg_to_m_lon * deg_to_m_lat
    
    # Convert all face areas to characteristic lengths
    face_areas_m2 = face_areas * deg_to_m2
    characteristic_lengths_all = np.sqrt(face_areas_m2)
    
    # Calculate original background resolution within analysis polygon for color limits
    original_face_areas = ugrid_original.grid.area
    original_face_areas_m2 = original_face_areas * deg_to_m2
    original_characteristic_lengths = np.sqrt(original_face_areas_m2)
    buffer_path_original = mpath.Path(np.array(polygons[0]))  # Use original sandpit polygon
    original_faces_in_buffer = buffer_path_original.contains_points(ugrid_original.grid.face_coordinates)
    if np.any(original_faces_in_buffer):
        background_resolution = np.median(original_characteristic_lengths[original_faces_in_buffer])
    else:
        background_resolution = np.median(original_characteristic_lengths)  # Fallback
    
    min_resolution = characteristic_lengths_all.min()
    
    print("Background resolution in analysis area:", round(background_resolution, 1), "m")
    print("Minimum resolution found:", round(min_resolution, 1), "m")
    
    # Buffer statistics
    if len(faces_indices_in_buffer) > 0:
        characteristic_lengths_buffer = characteristic_lengths_all[faces_indices_in_buffer]
        print("\nRESOLUTION WITHIN BUFFER POLYGON:")
        print("Characteristic length - Min:", round(characteristic_lengths_buffer.min(), 1), "m, Max:", round(characteristic_lengths_buffer.max(), 1), "m")
        print("Mean:", round(characteristic_lengths_buffer.mean(), 1), "m, Median:", round(np.median(characteristic_lengths_buffer), 1), "m")
    else:
        print("Warning: No faces found in buffer area")
        characteristic_lengths_buffer = []
    
    # Get smoothness values from MeshKernel
    smoothness_data = mk_object.mesh2d_get_smoothness()
    smoothness_values = smoothness_data.values
    valid_smoothness = smoothness_values[smoothness_values != -999.0]
    
    if len(valid_smoothness) > 0:
        print("\nSMOOTHNESS:")
        print("Range:", round(valid_smoothness.min(), 3), "-", round(valid_smoothness.max(), 3))
        print("Mean:", round(valid_smoothness.mean(), 3))
        
        # Check recommended threshold (1.4)
        exceed_recommended_smoothness = valid_smoothness > 1.4
        if np.any(exceed_recommended_smoothness):
            print("Warning:", np.sum(exceed_recommended_smoothness), "edges exceed recommended smoothness (>1.4)")
        
        # Check critical threshold (5.0)
        exceed_critical_smoothness = valid_smoothness > 5.0
        if np.any(exceed_critical_smoothness):
            print("Critical:", np.sum(exceed_critical_smoothness), "edges exceed critical smoothness (>5.0)")
        
        print("Targets: 1.2 (area of interest), 1.4 (recommended max), 5.0 (critical max)")
    else:
        print("Warning: No valid smoothness data available")
    
    # Get orthogonality values from MeshKernel
    orthogonality_data = mk_object.mesh2d_get_orthogonality()
    orthogonality_values = orthogonality_data.values
    valid_orthogonality = orthogonality_values[orthogonality_values != -999.0]
    
    if len(valid_orthogonality) > 0:
        print("\nORTHOGONALITY:")
        print("Range:", round(valid_orthogonality.min(), 4), "-", round(valid_orthogonality.max(), 4))
        print("Mean:", round(valid_orthogonality.mean(), 4))
        
        # Check recommended threshold (0.01)
        exceed_recommended_ortho = valid_orthogonality > 0.01
        if np.any(exceed_recommended_ortho):
            print("Warning:", np.sum(exceed_recommended_ortho), "edges exceed recommended orthogonality (>0.01)")
        
        # Check critical threshold (0.5)
        exceed_critical_ortho = valid_orthogonality > 0.5
        if np.any(exceed_critical_ortho):
            print("Critical:", np.sum(exceed_critical_ortho), "edges exceed critical orthogonality (>0.5)")
        
        print("Targets: ≤0.01 (recommended), ≤0.5 (critical threshold)")
    else:
        print("Warning: No valid orthogonality data available")
    
    return {
        'ugrid_refined': ugrid_refined,
        'characteristic_lengths_all': characteristic_lengths_all,
        'characteristic_lengths_buffer': characteristic_lengths_buffer,
        'smoothness_values': smoothness_values,
        'valid_smoothness': valid_smoothness,
        'orthogonality_values': orthogonality_values,
        'valid_orthogonality': valid_orthogonality,
        'background_resolution': background_resolution,
        'min_resolution': min_resolution,
        'mesh2d': mk_object.mesh2d_get()
    }


def plot_grid_quality(quality_data, all_refinement_polygons, target_resolution):
    """
    Create 6-panel visualization of grid quality metrics.
    
    Parameters
    ----------
    quality_data : dict
        Dictionary from analyze_grid_quality
    all_refinement_polygons : list
        Refinement polygons for axis limits
    target_resolution : float
        Target resolution in meters
    """
    # Extract data from quality dictionary
    ugrid_refined = quality_data['ugrid_refined']
    characteristic_lengths_all = quality_data['characteristic_lengths_all']
    characteristic_lengths_buffer = quality_data['characteristic_lengths_buffer']
    smoothness_values = quality_data['smoothness_values']
    valid_smoothness = quality_data['valid_smoothness']
    orthogonality_values = quality_data['orthogonality_values']
    valid_orthogonality = quality_data['valid_orthogonality']
    background_resolution = quality_data['background_resolution']
    min_resolution = quality_data['min_resolution']
    mesh2d = quality_data['mesh2d']
    
    # Set plot limits
    all_polygons = [poly for step_polys in all_refinement_polygons for poly in step_polys]
    x_min = min(np.min(np.array(poly)[:, 0]) for poly in all_polygons)
    x_max = max(np.max(np.array(poly)[:, 0]) for poly in all_polygons)
    y_min = min(np.min(np.array(poly)[:, 1]) for poly in all_polygons)
    y_max = max(np.max(np.array(poly)[:, 1]) for poly in all_polygons)
    
    # Create 6-panel visualization (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # Row 1: Resolution
    # Left: Map with resolution colored (logarithmic scale)
    ax1_map = axes[0, 0]
    ugrid_refined['resolution'] = ('mesh2d_nFaces', characteristic_lengths_all)
    
    # Use logarithmic normalization for better visualization of resolution differences
    norm = colors.LogNorm(vmin=min_resolution, vmax=background_resolution)
    im1 = ugrid_refined['resolution'].ugrid.plot(ax=ax1_map, cmap='jet', add_colorbar=False, norm=norm)
    cbar1 = plt.colorbar(im1, ax=ax1_map)
    cbar1.set_label('Resolution [m] (log scale)')
    
    ax1_map.set_xlim(x_min - 0.1, x_max + 0.1)
    ax1_map.set_ylim(y_min - 0.1, y_max + 0.1)
    ax1_map.set_xlabel('Longitude [degrees]')
    ax1_map.set_ylabel('Latitude [degrees]')
    ax1_map.set_title('Grid Resolution (sqrt of face areas)')
    ax1_map.grid(True, alpha=0.3)
    ax1_map.set_aspect('equal')
    
    # Right: Resolution histogram
    ax1_hist = axes[0, 1]
    if len(characteristic_lengths_buffer) > 0:
        ax1_hist.hist(characteristic_lengths_buffer, bins=20, alpha=0.7, color='blue', density=True)
        ax1_hist.axvline(target_resolution, color='red', linestyle='--', label='Target: ' + str(target_resolution) + 'm')
        ax1_hist.set_xlabel('Characteristic Length [m]')
        ax1_hist.set_ylabel('Density')
        ax1_hist.set_title('Resolution Distribution (Buffer Area)')
        ax1_hist.legend()
        ax1_hist.set_yscale('log')
        ax1_hist.grid(True, alpha=0.3)
    else:
        ax1_hist.text(0.5, 0.5, 'No resolution data\navailable', ha='center', va='center', transform=ax1_hist.transAxes)
        ax1_hist.set_title('Resolution Distribution')
    
    # Row 2: Smoothness
    # Left: Map with smoothness colored
    ax2_map = axes[1, 0]
    
    # Convert smoothness from edge data to face data for consistent plotting
    smoothness_edge_data = smoothness_values.copy()
    smoothness_edge_data[smoothness_edge_data == -999.0] = np.nan
    
    # Create temporary edge data array
    ugrid_refined['smoothness_edges'] = (ugrid_refined.grid.edge_dimension, smoothness_edge_data)
    # Convert to face data
    smoothness_face_data = ugrid_refined['smoothness_edges'].ugrid.to_face().mean(dim="nmax", keep_attrs=True)
    ugrid_refined['smoothness'] = smoothness_face_data
    
    # Plot using ugrid on faces
    im2 = ugrid_refined['smoothness'].ugrid.plot(ax=ax2_map, cmap='jet', add_colorbar=False, vmin=1.0, vmax=5.0)
    cbar2 = plt.colorbar(im2, ax=ax2_map)
    cbar2.set_label('Smoothness')
    
    # Add red crosses only for critical threshold (>5.0)
    exceed_critical_smoothness = (smoothness_values > 5.0) & (smoothness_values != -999.0)
    if np.any(exceed_critical_smoothness):
        edge_nodes = mesh2d.edge_nodes.reshape(-1, 2)
        edge_coords = np.column_stack([
            (mesh2d.node_x[edge_nodes[:, 0]] + mesh2d.node_x[edge_nodes[:, 1]]) / 2,
            (mesh2d.node_y[edge_nodes[:, 0]] + mesh2d.node_y[edge_nodes[:, 1]]) / 2
        ])
        exceed_coords = edge_coords[exceed_critical_smoothness]
        ax2_map.scatter(exceed_coords[:, 0], exceed_coords[:, 1], c='red', marker='x', s=20, alpha=0.8)
    
    ax2_map.set_xlim(x_min - 0.1, x_max + 0.1)
    ax2_map.set_ylim(y_min - 0.1, y_max + 0.1)
    ax2_map.set_xlabel('Longitude [degrees]')
    ax2_map.set_ylabel('Latitude [degrees]')
    ax2_map.set_title('Grid Smoothness')
    ax2_map.grid(True, alpha=0.3)
    ax2_map.set_aspect('equal')
    
    # Right: Smoothness histogram
    ax2_hist = axes[1, 1]
    if len(valid_smoothness) > 0:
        ax2_hist.hist(valid_smoothness, bins=30, alpha=0.7, color='green')
        ax2_hist.axvline(1.2, color='blue', linestyle='--', label='Target: 1.2 (AOI)')
        ax2_hist.axvline(1.4, color='orange', linestyle='--', label='Recommended: 1.4')
        ax2_hist.axvline(5.0, color='red', linestyle='--', label='Critical: 5.0')
        ax2_hist.set_xlabel('Smoothness')
        ax2_hist.set_ylabel('Frequency')
        ax2_hist.set_title('Smoothness Distribution')
        ax2_hist.legend()
        ax2_hist.set_yscale('log')
        ax2_hist.grid(True, alpha=0.3)
    else:
        ax2_hist.text(0.5, 0.5, 'No smoothness data\navailable', ha='center', va='center', transform=ax2_hist.transAxes)
        ax2_hist.set_title('Smoothness Distribution')
    
    # Row 3: Orthogonality
    # Left: Map with orthogonality colored
    ax3_map = axes[2, 0]
    
    # Convert orthogonality from edge data to face data for consistent plotting
    orthogonality_edge_data = orthogonality_values.copy()
    orthogonality_edge_data[orthogonality_edge_data == -999.0] = np.nan
    
    # Create temporary edge data array
    ugrid_refined['orthogonality_edges'] = (ugrid_refined.grid.edge_dimension, orthogonality_edge_data)
    # Convert to face data
    orthogonality_face_data = ugrid_refined['orthogonality_edges'].ugrid.to_face().mean(dim="nmax", keep_attrs=True)
    ugrid_refined['orthogonality'] = orthogonality_face_data
    
    # Plot using ugrid on faces
    im3 = ugrid_refined['orthogonality'].ugrid.plot(ax=ax3_map, cmap='jet', add_colorbar=False, vmin=0.0, vmax=0.5)
    cbar3 = plt.colorbar(im3, ax=ax3_map)
    cbar3.set_label('Orthogonality')
    
    # Add red crosses only for critical threshold (>0.5)
    exceed_critical_orthogonality = (orthogonality_values > 0.5) & (orthogonality_values != -999.0)
    if np.any(exceed_critical_orthogonality):
        edge_nodes = mesh2d.edge_nodes.reshape(-1, 2)
        edge_coords = np.column_stack([
            (mesh2d.node_x[edge_nodes[:, 0]] + mesh2d.node_x[edge_nodes[:, 1]]) / 2,
            (mesh2d.node_y[edge_nodes[:, 0]] + mesh2d.node_y[edge_nodes[:, 1]]) / 2
        ])
        exceed_coords = edge_coords[exceed_critical_orthogonality]
        ax3_map.scatter(exceed_coords[:, 0], exceed_coords[:, 1], c='red', marker='x', s=20, alpha=0.8)
    
    ax3_map.set_xlim(x_min - 0.1, x_max + 0.1)
    ax3_map.set_ylim(y_min - 0.1, y_max + 0.1)
    ax3_map.set_xlabel('Longitude [degrees]')
    ax3_map.set_ylabel('Latitude [degrees]')
    ax3_map.set_title('Grid Orthogonality')
    ax3_map.grid(True, alpha=0.3)
    ax3_map.set_aspect('equal')
    
    # Right: Orthogonality histogram
    ax3_hist = axes[2, 1]
    if len(valid_orthogonality) > 0:
        ax3_hist.hist(valid_orthogonality, bins=30, alpha=0.7, color='purple')
        ax3_hist.axvline(0.01, color='orange', linestyle='--', label='Recommended: ≤0.01')
        ax3_hist.axvline(0.5, color='red', linestyle='--', label='Critical: ≤0.5')
        ax3_hist.set_xlabel('Orthogonality')
        ax3_hist.set_ylabel('Frequency')
        ax3_hist.set_title('Orthogonality Distribution')
        ax3_hist.legend()
        ax3_hist.grid(True, alpha=0.3)
        
        # Fix divide by zero warning by checking for valid range
        if len(valid_orthogonality) > 1 and valid_orthogonality.min() > 0:
            if valid_orthogonality.max() / valid_orthogonality.min() > 100:
                ax3_hist.set_yscale('log')
        elif len(valid_orthogonality) > 1:
            # Use log scale if we have a wide range but avoid divide by zero
            ax3_hist.set_yscale('log')
    else:
        ax3_hist.text(0.5, 0.5, 'No orthogonality data\navailable', ha='center', va='center', transform=ax3_hist.transAxes)
        ax3_hist.set_title('Orthogonality Distribution')
    
    plt.tight_layout()
    plt.show()
    
    print("\nGrid quality analysis complete")