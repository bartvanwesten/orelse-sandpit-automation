"""
Bed level modification for sandpit excavation in D-Flow FM restart files.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.spatial.distance import cdist


def get_transition_zones(coordinates, polygons, transition_distance_m):
    """Calculate transition zones around polygons for smooth bed level changes."""
    # Convert to degrees
    lat_approx = np.mean(coordinates[:, 1])
    meters_per_degree = 111320 * np.cos(np.radians(lat_approx))
    transition_distance_deg = transition_distance_m / meters_per_degree
    
    # Convert to GeoDataFrame
    points_gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in coordinates])
    
    # Combine polygons
    shapely_polygons = [Polygon(poly) for poly in polygons]
    combined_poly = shapely_polygons[0] if len(shapely_polygons) == 1 else gpd.GeoSeries(shapely_polygons).union_all()
    
    # Create zones
    within_sandpit = points_gdf.geometry.within(combined_poly)
    inner_poly = combined_poly.buffer(-transition_distance_deg)
    
    if hasattr(inner_poly, 'area') and inner_poly.area > 0:
        within_inner = points_gdf.geometry.within(inner_poly)
    else:
        within_inner = np.zeros(len(points_gdf), dtype=bool)
    
    in_transition = within_sandpit & (~within_inner)
    in_full_excavation = within_inner
    
    # Calculate transition factors
    transition_factors = np.zeros(len(coordinates))
    
    if in_transition.any():
        boundary = combined_poly.boundary
        if boundary.geom_type == 'LineString':
            boundary_coords = np.array(boundary.coords)
        else:
            boundary_coords = np.vstack([np.array(geom.coords) for geom in boundary.geoms])
        
        transition_points = coordinates[in_transition]
        distances_deg = cdist(transition_points, boundary_coords)
        min_distances_deg = np.min(distances_deg, axis=1)
        min_distances_m = min_distances_deg * meters_per_degree
        
        factors = np.clip(min_distances_m / transition_distance_m, 0, 1)
        transition_factors[in_transition] = factors
    
    transition_factors[in_full_excavation] = 1.0
    
    return transition_factors


def modify_bed_levels(datasets, polygons, dig_depth, slope):
    """
    Modify bed levels in restart datasets for sandpit excavation.
    
    Parameters
    ----------
    datasets : list
        List of (partition_number, dataset) tuples
    polygons : list
        List of sandpit polygon coordinate lists
    dig_depth : float
        Excavation depth in meters
    slope : float
        Transition slope ratio
        
    Returns
    -------
    list
        List of (partition_number, modified_dataset) tuples
    """
    transition_distance_m = dig_depth / slope if slope > 0 else 0
    modified_datasets = []
    
    for partition_num, dataset in datasets:
        coordinates = np.column_stack([dataset.FlowElem_xcc.values, dataset.FlowElem_ycc.values])
        transition_factors = get_transition_zones(coordinates, polygons, transition_distance_m)
        
        bed_levels = dataset['FlowElem_bl'].values.copy()
        depth_changes = transition_factors * dig_depth
        
        if len(bed_levels.shape) == 2:
            for t in range(bed_levels.shape[0]):
                bed_levels[t] -= depth_changes
        else:
            bed_levels -= depth_changes
        
        dataset_modified = dataset.copy()
        dataset_modified['FlowElem_bl'] = (dataset['FlowElem_bl'].dims, bed_levels)
        
        modified_datasets.append((partition_num, dataset_modified))
    
    return modified_datasets