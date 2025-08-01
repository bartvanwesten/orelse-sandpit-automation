"""
Utility functions for grid resolution calculations and refinement planning.
"""

import numpy as np
import matplotlib.path as mpath
from collections import defaultdict


def degrees_to_meters(lat_deg):
    """
    Convert degrees to meters at given latitude.
    
    Parameters
    ----------
    lat_deg : float
        Latitude in degrees
        
    Returns
    -------
    tuple
        (lon_to_m, lat_to_m) conversion factors for longitude and latitude
    """
    earth_radius = 6371000  # Earth radius in meters
    lat_rad = np.radians(lat_deg)
    
    # 1 degree longitude in meters at given latitude
    lon_to_m = earth_radius * np.cos(lat_rad) * np.pi / 180
    # 1 degree latitude in meters (constant)
    lat_to_m = earth_radius * np.pi / 180
    
    return lon_to_m, lat_to_m


def analyze_grid_resolution_in_polygons(ugrid, polygons):
    """
    Analyze grid resolution within polygons by finding nodes inside polygons
    and computing horizontal/vertical resolution based on actual edge connectivity.
    
    Parameters
    ----------
    ugrid : xugrid dataset
        Grid dataset with node coordinates and edge connectivity
    polygons : list
        List of polygon coordinates [[x1,y1],[x2,y2],...]
        
    Returns
    -------
    list
        List of node resolution data
    """
    # Get node coordinates and edge connectivity
    node_coords = ugrid.grid.node_coordinates
    edge_nodes = ugrid.grid.edge_node_connectivity
    
    nodes_in_polygons = []
    
    # Process each polygon
    for poly_idx, polygon in enumerate(polygons):
        # Create matplotlib path for point-in-polygon test
        poly_path = mpath.Path(np.array(polygon))
        
        # Find nodes inside this polygon
        inside_mask = poly_path.contains_points(node_coords)
        inside_node_indices = np.where(inside_mask)[0]
        
        # For each node inside polygon, find connected edges and calculate spacings
        for node_idx in inside_node_indices:
            node_coord = node_coords[node_idx]
            
            # Find all edges connected to this node
            connected_edges = np.where((edge_nodes[:, 0] == node_idx) | (edge_nodes[:, 1] == node_idx))[0]
            
            if len(connected_edges) == 0:
                continue
            
            # Calculate edge lengths for connected edges
            h_spacings = []  # horizontal spacings
            v_spacings = []  # vertical spacings
            
            for edge_idx in connected_edges:
                # Get the two nodes of this edge
                node1_idx, node2_idx = edge_nodes[edge_idx]
                
                # Get coordinates of both nodes
                node1_coord = node_coords[node1_idx]
                node2_coord = node_coords[node2_idx]
                
                # Calculate horizontal and vertical differences
                lon_diff = abs(node2_coord[0] - node1_coord[0])
                lat_diff = abs(node2_coord[1] - node1_coord[1])
                
                # Convert to meters at the node's latitude
                center_lat = node_coord[1]
                lon_to_m, lat_to_m = degrees_to_meters(center_lat)
                
                h_spacing_m = lon_diff * lon_to_m
                v_spacing_m = lat_diff * lat_to_m
                
                h_spacings.append(h_spacing_m)
                v_spacings.append(v_spacing_m)
            
            # Use max spacing as representative for this node
            if h_spacings and v_spacings:
                h_res_m = np.max(h_spacings)
                v_res_m = np.max(v_spacings)
                
                nodes_in_polygons.append({
                    'polygon_idx': poly_idx,
                    'node_idx': node_idx,
                    'h_resolution_m': h_res_m,
                    'v_resolution_m': v_res_m,
                    'node_lon': node_coord[0],
                    'node_lat': node_coord[1]
                })
    
    return nodes_in_polygons


def analyze_resolution_patterns(nodes_data):
    """
    Analyze resolution patterns and return summary statistics.
    
    Parameters
    ----------
    nodes_data : list
        List of node resolution data from analyze_grid_resolution_in_polygons
        
    Returns
    -------
    tuple
        (h_resolution_counts, v_resolution_counts) dictionaries
    """
    # Group by horizontal resolution (rounded to nearest 10m)
    h_resolution_counts = defaultdict(int)
    v_resolution_counts = defaultdict(int)
    
    for node in nodes_data:
        h_res = round(node['h_resolution_m'] / 10) * 10  # Round to nearest 10m
        v_res = round(node['v_resolution_m'] / 10) * 10  # Round to nearest 10m
        
        h_resolution_counts[h_res] += 1
        v_resolution_counts[v_res] += 1
    
    return h_resolution_counts, v_resolution_counts


def compute_refinement_steps(ugrid, target_resolution, polygons):
    """
    Compute number of refinement steps and envelope sizes needed.
    
    Parameters
    ----------
    ugrid : xugrid dataset
        Grid dataset with coordinates
    target_resolution : float
        Target resolution in meters
    polygons : list
        List of polygon coordinates for detailed analysis
        
    Returns
    -------
    dict
        Dictionary containing refinement parameters
    """
    # Analyze actual grid resolution in polygons
    nodes_data = analyze_grid_resolution_in_polygons(ugrid, polygons)
    h_resolution_counts, v_resolution_counts = analyze_resolution_patterns(nodes_data)
    
    # Find the maximum (coarsest) resolution from both horizontal and vertical
    max_h_res = max(h_resolution_counts.keys())
    max_v_res = max(v_resolution_counts.keys())
    current_spacing_m = max(max_h_res, max_v_res)
    
    # Calculate number of refinement steps needed
    # Each refinement step halves the resolution
    refinement_ratio = current_spacing_m / target_resolution
    n_refinement_steps = int(np.ceil(np.log2(refinement_ratio)))
    
    # Compute envelope sizes for each refinement step
    envelope_sizes = []
    current_envelope = current_spacing_m
    
    for step in range(n_refinement_steps + 1):  # Include initial step
        envelope_sizes.append(current_envelope)
        if step < n_refinement_steps:
            current_envelope = current_envelope / 2
    
    print(f"Grid analysis: Current {current_spacing_m:.0f}m → Target {target_resolution}m in {n_refinement_steps} steps")
    
    return {
        'n_steps': n_refinement_steps,
        'envelope_sizes_m': envelope_sizes,
        'target_resolution': target_resolution,
        'current_spacing_m': current_spacing_m
    }


def apply_casulli_refinement(mk_object, all_refinement_polygons):
    """
    Apply Casulli refinement to the grid using the refinement polygons.
    
    Parameters
    ----------
    mk_object : meshkernel object
        MeshKernel object to refine
    all_refinement_polygons : list
        List of refinement polygon sets (from coarse to fine)
    """
    from meshkernel import GeometryList
    
    print("Applying Casulli refinement...")
    
    # Perform Casulli refinement from outside to inside (coarse to fine)
    for step, step_polygons in enumerate(all_refinement_polygons):
        for i, polygon in enumerate(step_polygons):
            pol = np.array(polygon)
            
            # Ensure polygon is closed by adding first point as last if needed
            if not np.allclose(pol[0], pol[-1]):
                pol = np.vstack([pol, pol[0]])
            
            poly_geolist = GeometryList(x_coordinates=pol[:,0], y_coordinates=pol[:,1])
            mk_object.mesh2d_casulli_refinement_on_polygon(poly_geolist)
        
        print(f"  Level {step+1}: refined {len(step_polygons)} zones")


def print_refinement_summary(polygons, all_refinement_polygons, envelope_sizes_m, n_steps, buffer_polygons):
    """
    Print minimal summary of the refinement process.
    
    Parameters
    ----------
    polygons : list
        Original sandpit polygons
    all_refinement_polygons : list
        Final refinement polygons after merging
    envelope_sizes_m : list
        Target resolutions for each step
    n_steps : int
        Number of refinement steps
    buffer_polygons : list
        Buffer polygons
    """
    total_refinement_zones = sum(len(step_polys) for step_polys in all_refinement_polygons)
    print(f"\nRefinement summary:")
    print(f"  {len(polygons)} sandpit(s) → {total_refinement_zones} refinement zones + {len(buffer_polygons)} buffer zones")
    print(f"  Target resolution: {envelope_sizes_m[-1]:.0f}m achieved")
