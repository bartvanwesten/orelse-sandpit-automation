"""
Utility functions for polygon operations in sandpit refinement workflow.
"""

import numpy as np
from datetime import datetime
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import os


class InteractivePolygonDrawer:
    def __init__(self, ugrid_dataset, nc_path, plot_bathymetry):
        """
        Initialize InteractivePolygonDrawer with dfm_tools dataset
        
        Parameters
        ----------
        ugrid_dataset : xugrid.UgridDataset
            The dfm_tools ugrid dataset
        nc_path : str
            Path to netCDF file (for backup loading if needed)
        plot_bathymetry : bool
            Whether to plot bathymetry in the background
        """
        self.ugrid = ugrid_dataset
        self.nc_path = nc_path
        self.polygons = []
        self.current_polygon = []
        self.fig = None
        self.ax = None
        self.plot_bathymetry = plot_bathymetry
        
    def draw_polygons(self):
        """Interactive polygon drawing interface with dfm_tools background"""
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        

        # print("Rendering bathymetry background...")
        # if 'NetNode_z' in self.ugrid:
        #     self.ugrid['NetNode_z'].ugrid.plot(ax=self.ax, cmap='terrain', alpha=0.7, add_colorbar=False)
        # elif 'mesh2d_node_z' in self.ugrid:
        #     self.ugrid['mesh2d_node_z'].ugrid.plot(ax=self.ax, cmap='terrain', alpha=0.7, add_colorbar=False)

        # Plot grid using dfm_tools - much more efficient
        if self.plot_bathymetry:
            print("Rendering grid background...")
            if 'NetNode_z' in self.ugrid:
                self.ugrid['NetNode_z'].ugrid.plot(ax=self.ax, cmap='ocean', alpha=0.7, add_colorbar=True, vmin=-40, vmax=0, linewidth=0)
            elif 'mesh2d_node_z' in self.ugrid:
                self.ugrid['mesh2d_node_z'].ugrid.plot(ax=self.ax, cmap='ocean', alpha=0.7, add_colorbar=True, vmin=-40, vmax=0, linewidth=0)

        self.ugrid.grid.plot(ax=self.ax, linewidth=0.5, color='black', alpha=0.3)
        
        self.ax.set_xlabel('Longitude [degrees]')
        self.ax.set_ylabel('Latitude [degrees]')
        self.ax.set_title('LEFT CLICK: pan/zoom | RIGHT CLICK: add point | ENTER: finish polygon | Close window when done')
        self.ax.grid(False)  # Turn off default grid lines
        self.ax.set_aspect('equal')
        
        # Tight layout with minimal margins
        plt.tight_layout()
        plt.subplots_adjust(left=0.08, right=0.82, top=0.93, bottom=0.1)  # More space for legend
        
        # Connect mouse and keyboard events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        
        print("Interactive plot ready. Right-click to add points, Enter to finish polygon.")
        plt.show(block=True)  # Block until window is closed
        
        return self.polygons
    
    def _on_close(self, event):
        """Handle window close event"""
        if len(self.polygons) > 0:
            print(f"\nSession complete: {len(self.polygons)} polygons created")
        else:
            print("\nNo polygons were created")
    
    def _update_legend(self):
        """Update the legend with current polygons"""
        if len(self.polygons) > 0:
            # Create legend handles
            handles = []
            labels = []
            colors = plt.cm.tab10.colors
            
            for i in range(len(self.polygons)):
                color = colors[i % len(colors)]
                # Create a patch for the legend
                patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3, edgecolor=color)
                handles.append(patch)
                labels.append(f'SandPit_{i+1:03d}')
            
            self.ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            # Remove legend if no polygons
            legend = self.ax.get_legend()
            if legend:
                legend.remove()
    
    def _on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'enter':
            if len(self.current_polygon) >= 3:
                # Get color for this polygon
                colors = plt.cm.tab10.colors  # Default matplotlib color cycle
                color = colors[len(self.polygons) % len(colors)]
                
                # Close polygon
                first_point = self.current_polygon[0]
                last_point = self.current_polygon[-1]
                self.ax.plot([last_point[0], first_point[0]], 
                           [last_point[1], first_point[1]], color=color, linewidth=2)
                
                # Fill polygon
                poly_array = np.array(self.current_polygon)
                self.ax.fill(poly_array[:, 0], poly_array[:, 1], 
                           color=color, alpha=0.3)
                
                # Save polygon
                self.polygons.append(self.current_polygon.copy())
                print(f"Polygon {len(self.polygons)} completed with {len(self.current_polygon)} vertices")
                
                # Update legend
                self._update_legend()
                
                # Save immediately after each polygon
                temp_file = 'temp_sandpits.pol'
                save_pol_file(self.polygons, temp_file)
                print(f"  -> Saved to {temp_file}")
                
                # Reset for next polygon
                self.current_polygon = []
                
                self.fig.canvas.draw()
            else:
                print("Need at least 3 points to finish polygon")
    
    def _on_click(self, event):
        """Handle mouse clicks for polygon drawing"""
        if event.inaxes != self.ax:
            return
            
        if event.button == 3:  # Right click - add point
            # Get color for current polygon being drawn
            colors = plt.cm.tab10.colors
            color = colors[len(self.polygons) % len(colors)]
            
            self.current_polygon.append([event.xdata, event.ydata])
            self.ax.plot(event.xdata, event.ydata, 'o', color=color, markersize=6)
            
            # Draw line to previous point
            if len(self.current_polygon) > 1:
                prev_point = self.current_polygon[-2]
                self.ax.plot([prev_point[0], event.xdata], 
                           [prev_point[1], event.ydata], color=color, linewidth=2)
            
            print(f"Added point {len(self.current_polygon)}: ({event.xdata:.6f}, {event.ydata:.6f})")
            self.fig.canvas.draw()


def load_pol_file(file_path):
    """Load polygons from RGFGRID .pol file format"""
    polygons = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip comment lines
        if line.startswith('*') or not line:
            i += 1
            continue
        
        # This should be a polygon name
        polygon_name = line
        i += 1
        
        if i >= len(lines):
            break
            
        # Next line should contain number of points and columns
        size_line = lines[i].strip().split()
        n_points = int(size_line[0])
        i += 1
        
        # Read polygon coordinates
        polygon = []
        for j in range(n_points):
            if i + j < len(lines):
                coords = lines[i + j].strip().split()
                x, y = float(coords[0]), float(coords[1])
                polygon.append([x, y])
        
        if polygon:
            polygons.append(polygon)
        
        i += n_points
    
    return polygons


def save_pol_file(polygons, file_path):
    """Save polygons to RGFGRID .pol file format"""
    with open(file_path, 'w') as f:
        # Write header
        f.write("*\n")
        f.write(f"* Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("*\n")
        
        # Write each polygon
        for i, polygon in enumerate(polygons):
            polygon_name = f"SandPit_{i+1:03d}"
            
            # Ensure polygon is closed
            if len(polygon) > 0:
                first_point = polygon[0]
                last_point = polygon[-1]
                
                # Add first point as last if not already closed
                if abs(first_point[0] - last_point[0]) > 1e-10 or abs(first_point[1] - last_point[1]) > 1e-10:
                    polygon = polygon + [first_point]
            
            f.write(f"{polygon_name}\n")
            f.write(f"{len(polygon)} 2\n")
            
            for point in polygon:
                f.write(f"{point[0]:.10E} {point[1]:.10E}\n")


def expand_polygon_outward(polygon, expansion_distance_m, center_lat):
    """
    Expand a polygon outward using Shapely's buffer method.
    
    Parameters
    ----------
    polygon : list
        List of [lon, lat] coordinates defining the polygon
    expansion_distance_m : float
        Distance to expand outward in meters
    center_lat : float
        Representative latitude for degree-to-meter conversion
        
    Returns
    -------
    list
        Expanded polygon coordinates
    """
    # Convert expansion distance from meters to degrees
    earth_radius = 6371000  # Earth radius in meters
    lat_rad = np.radians(center_lat)
    lon_to_m = earth_radius * np.cos(lat_rad) * np.pi / 180
    lat_to_m = earth_radius * np.pi / 180
    avg_m_per_deg = (lon_to_m + lat_to_m) / 2
    expansion_deg = expansion_distance_m / avg_m_per_deg
    
    # Convert to Shapely polygon
    polygon_array = np.array(polygon)
    
    # Remove duplicate first/last point if present
    if len(polygon_array) > 1 and np.allclose(polygon_array[0], polygon_array[-1]):
        polygon_array = polygon_array[:-1]
    
    # Create Shapely polygon
    shapely_poly = Polygon(polygon_array)
    
    # Fix invalid polygons
    if not shapely_poly.is_valid:
        shapely_poly = shapely_poly.buffer(0)
    
    # Apply buffer expansion
    expanded_poly = shapely_poly.buffer(expansion_deg, join_style=2, mitre_limit=10.0)
    
    # Handle MultiPolygon case (take the largest polygon)
    if hasattr(expanded_poly, 'geoms'):
        # MultiPolygon - take the largest one
        expanded_poly = max(expanded_poly.geoms, key=lambda p: p.area)
    
    # Extract coordinates
    coords = list(expanded_poly.exterior.coords)
    # Remove the duplicate last point that Shapely adds
    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    
    return coords


def line_intersection(p1, p2, p3, p4):
    """
    Find intersection point of two lines defined by points (p1,p2) and (p3,p4).
    
    Parameters
    ----------
    p1, p2 : array-like
        Points defining first line
    p3, p4 : array-like
        Points defining second line
        
    Returns
    -------
    array or None
        Intersection point or None if lines are parallel
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Calculate denominators
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-12:  # Lines are parallel
        return None
    
    # Calculate intersection point
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    intersection_x = x1 + t * (x2 - x1)
    intersection_y = y1 + t * (y2 - y1)
    
    return np.array([intersection_x, intersection_y])


def generate_refinement_polygons(polygons, refinement_params, buffer_around_sandpit, N):
    """
    Generate refinement polygons for each step with overlap merging.
    
    Parameters
    ----------
    polygons : list
        Original sandpit polygons
    refinement_params : dict
        Refinement parameters from compute_refinement_steps
    buffer_around_sandpit : float
        Buffer distance around sandpit in meters
    N : int
        Number of transition cells
        
    Returns
    -------
    tuple
        (all_refinement_polygons, all_original_polygons, buffer_polygons, expansions)
    """
    # Get refinement parameters
    n_steps = refinement_params['n_steps']
    envelope_sizes_m = refinement_params['envelope_sizes_m']
    current_spacing_m = refinement_params['current_spacing_m']
    
    print(f"Generating {n_steps} refinement levels: {current_spacing_m:.0f}m → {refinement_params['target_resolution']}m")
    
    # Calculate expansion distances working from innermost to outermost
    expansions = []
    cumulative_expansion = buffer_around_sandpit  # Start with buffer distance
    
    # Work from finest to coarsest (inside to outside)
    for step in range(n_steps):
        step_index = n_steps - step  # n_steps, n_steps-1, ..., 1
        current_res = envelope_sizes_m[step_index]
        
        # Transition width for this refinement level
        transition_width = (N + 2) * current_res
        cumulative_expansion += transition_width
        expansions.append(cumulative_expansion)
    
    # Reverse to get outermost first (for Casulli application order)
    expansions.reverse()
    
    # Generate refinement polygons with overlap checking and merging
    all_refinement_polygons = []
    all_original_polygons = []
    
    # Create polygons in order: outermost (coarse) to innermost (fine)
    for step_idx, expansion_distance in enumerate(expansions):
        step_polygons = []
        for i, polygon in enumerate(polygons):  # Always start from original sandpit
            polygon_array = np.array(polygon)
            center_lat = np.mean(polygon_array[:, 1])
            expanded_polygon = expand_polygon_outward(polygon, expansion_distance, center_lat)
            step_polygons.append(expanded_polygon)
        
        # Store original polygons before merging
        all_original_polygons.append(step_polygons)
        
        # Check for overlaps and merge if necessary
        # Convert to Shapely polygons for overlap detection
        shapely_polygons = []
        for poly in step_polygons:
            try:
                shapely_poly = Polygon(poly)
                if shapely_poly.is_valid:
                    shapely_polygons.append(shapely_poly)
                else:
                    # Try to fix invalid polygon
                    shapely_poly = shapely_poly.buffer(0)
                    shapely_polygons.append(shapely_poly)
            except:
                # Skip invalid polygons
                print(f"Warning: Skipping invalid polygon in step {step_idx+1}")
                continue
        
        if len(shapely_polygons) == 0:
            print(f"Warning: No valid polygons in step {step_idx+1}")
            continue
        
        # Find groups of overlapping polygons
        merged_groups = []
        processed = [False] * len(shapely_polygons)
        
        for i in range(len(shapely_polygons)):
            if processed[i]:
                continue
                
            # Start a new group with polygon i
            current_group = [i]
            processed[i] = True
            
            # Find all polygons that overlap with any polygon in the current group
            changed = True
            while changed:
                changed = False
                for j in range(len(shapely_polygons)):
                    if processed[j]:
                        continue
                    
                    # Check if polygon j overlaps with any polygon in current group
                    for group_idx in current_group:
                        if shapely_polygons[j].intersects(shapely_polygons[group_idx]):
                            current_group.append(j)
                            processed[j] = True
                            changed = True
                            break
            
            merged_groups.append(current_group)
        
        # Create final polygons based on merged groups
        merged_step_polygons = []
        
        for group in merged_groups:
            if len(group) == 1:
                # Single polygon, no merging needed
                idx = group[0]
                coords = list(shapely_polygons[idx].exterior.coords)
                merged_step_polygons.append(coords)
            else:
                # Multiple overlapping polygons, merge them
                polygons_to_merge = [shapely_polygons[idx] for idx in group]
                merged_polygon = unary_union(polygons_to_merge).convex_hull
                coords = list(merged_polygon.exterior.coords)
                merged_step_polygons.append(coords)
                print(f"  Step {step_idx+1}: merged {len(group)} overlapping polygons")
        
        all_refinement_polygons.append(merged_step_polygons)
        
        # Report results
        target_res = envelope_sizes_m[n_steps - step_idx]  # Outermost=coarsest, innermost=finest
        print(f"  Level {step_idx+1}: {len(step_polygons)} → {len(merged_step_polygons)} polygons @ {target_res:.0f}m")
    
    # Handle buffer polygons with overlap merging  
    original_buffer_polygons = []
    for i, polygon in enumerate(polygons):
        polygon_array = np.array(polygon)
        center_lat = np.mean(polygon_array[:, 1])
        expanded_polygon = expand_polygon_outward(polygon, buffer_around_sandpit, center_lat)
        original_buffer_polygons.append(expanded_polygon)
    
    # Check buffer overlaps using the same group-based approach
    shapely_buffer_polygons = []
    for poly in original_buffer_polygons:
        try:
            shapely_poly = Polygon(poly)
            if shapely_poly.is_valid:
                shapely_buffer_polygons.append(shapely_poly)
            else:
                shapely_poly = shapely_poly.buffer(0)
                shapely_buffer_polygons.append(shapely_poly)
        except:
            print("Warning: Skipping invalid buffer polygon")
            continue
    
    # Find groups of overlapping buffer polygons
    buffer_merged_groups = []
    buffer_processed = [False] * len(shapely_buffer_polygons)
    
    for i in range(len(shapely_buffer_polygons)):
        if buffer_processed[i]:
            continue
            
        # Start a new group with polygon i
        current_group = [i]
        buffer_processed[i] = True
        
        # Find all polygons that overlap with any polygon in the current group
        changed = True
        while changed:
            changed = False
            for j in range(len(shapely_buffer_polygons)):
                if buffer_processed[j]:
                    continue
                
                # Check if polygon j overlaps with any polygon in current group
                for group_idx in current_group:
                    if shapely_buffer_polygons[j].intersects(shapely_buffer_polygons[group_idx]):
                        current_group.append(j)
                        buffer_processed[j] = True
                        changed = True
                        break
        
        buffer_merged_groups.append(current_group)
    
    # Create final buffer polygons based on merged groups
    buffer_polygons = []
    
    for group in buffer_merged_groups:
        if len(group) == 1:
            # Single polygon, no merging needed
            idx = group[0]
            coords = list(shapely_buffer_polygons[idx].exterior.coords)
            buffer_polygons.append(coords)
        else:
            # Multiple overlapping polygons, merge them
            polygons_to_merge = [shapely_buffer_polygons[idx] for idx in group]
            merged_polygon = unary_union(polygons_to_merge).convex_hull
            coords = list(merged_polygon.exterior.coords)
            buffer_polygons.append(coords)
            print(f"  Buffer: merged {len(group)} overlapping polygons")
    
    print(f"Buffer polygons: {len(original_buffer_polygons)} → {len(buffer_polygons)} @ {buffer_around_sandpit}m")
    
    return all_refinement_polygons, all_original_polygons, buffer_polygons, expansions


def add_shape_to_polygon(input_pol_file, output_pol_file, shape_type, center_lon, center_lat, **kwargs):
    """
    Add geometric shapes to a polygon file
    
    Parameters
    ----------
    input_pol_file : str
        Path to input .pol file (will be created if doesn't exist)
    output_pol_file : str
        Path to output .pol file (will be created if doesn't exist)
    shape_type : str
        Type of shape: 'circle', 'rectangle', 'ellipse' (placeholder), 'triangle' (placeholder)
    center_lon : float
        Longitude of shape center
    center_lat : float
        Latitude of shape center
    **kwargs : dict
        Shape-specific parameters (all in degrees):
        - For 'circle': radius (in degrees)
        - For 'rectangle': width (in degrees), height (in degrees), rotation (in degrees, default=0)
        - For 'ellipse': major_axis, minor_axis, rotation (placeholders)
        - For 'triangle': side_length, rotation (placeholders)
    
    Returns
    -------
    None
    """
    # Load existing polygons if file exists
    if os.path.exists(input_pol_file):
        existing_polygons = load_pol_file(input_pol_file)
        print(f"📂 Loading existing file: {len(existing_polygons)} polygons found")
    else:
        existing_polygons = []
    
    print(f"📝 Creating output polygon file: {output_pol_file}")
    
    # Generate shape coordinates
    if shape_type.lower() == 'circle':
        radius_deg = kwargs.get('radius')
        if radius_deg is None:
            raise ValueError("Circle requires 'radius' parameter (in degrees)")
        
        # Generate circle points (64 points for smooth circle)
        n_points = kwargs.get('n_points', 64)
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        
        polygon_coords = []
        for angle in angles:
            lon = center_lon + radius_deg * np.cos(angle)
            lat = center_lat + radius_deg * np.sin(angle)
            polygon_coords.append([lon, lat])
        
        print(f"✅ Generated circle: center=({center_lon:.6f}, {center_lat:.6f}), radius={radius_deg}°")
    
    elif shape_type.lower() == 'rectangle':
        width_deg = kwargs.get('width')
        height_deg = kwargs.get('height')
        rotation_deg = kwargs.get('rotation', 0)
        
        if width_deg is None or height_deg is None:
            raise ValueError("Rectangle requires 'width' and 'height' parameters (in degrees)")
        
        # Define rectangle corners (before rotation)
        half_width = width_deg / 2
        half_height = height_deg / 2
        
        corners = [
            [-half_width, -half_height],  # bottom-left
            [half_width, -half_height],   # bottom-right
            [half_width, half_height],    # top-right
            [-half_width, half_height]    # top-left
        ]
        
        # Apply rotation if specified
        if rotation_deg != 0:
            rotation_rad = np.radians(rotation_deg)
            cos_rot = np.cos(rotation_rad)
            sin_rot = np.sin(rotation_rad)
            
            rotated_corners = []
            for dx, dy in corners:
                # Rotate point around origin
                new_dx = dx * cos_rot - dy * sin_rot
                new_dy = dx * sin_rot + dy * cos_rot
                rotated_corners.append([new_dx, new_dy])
            corners = rotated_corners
        
        # Translate to center position
        polygon_coords = []
        for dx, dy in corners:
            lon = center_lon + dx
            lat = center_lat + dy
            polygon_coords.append([lon, lat])
        
        print(f"✅ Generated rectangle: center=({center_lon:.6f}, {center_lat:.6f}), "
              f"size={width_deg}×{height_deg}°, rotation={rotation_deg}°")
    
    elif shape_type.lower() == 'ellipse':
        # Placeholder for future implementation
        raise NotImplementedError("Ellipse shape not yet implemented. Use 'circle' or 'rectangle'.")
    
    elif shape_type.lower() == 'triangle':
        # Placeholder for future implementation
        raise NotImplementedError("Triangle shape not yet implemented. Use 'circle' or 'rectangle'.")
    
    else:
        raise ValueError(f"Unknown shape_type '{shape_type}'. Available: 'circle', 'rectangle'")
    
    # Add new shape to existing polygons
    existing_polygons.append(polygon_coords)
    
    # Save updated polygon file
    save_pol_file(existing_polygons, output_pol_file)
    
    total_polygons = len(existing_polygons)
    print(f"💾 Saved {total_polygons} polygons to {output_pol_file}")
    print(f"   └── Added {shape_type}: SandPit_{total_polygons:03d}")
