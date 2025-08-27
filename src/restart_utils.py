"""
Utility functions for restart file analysis and interpolation in sandpit refinement workflow.
"""

import numpy as np
import xarray as xr
from datetime import datetime
from scipy.spatial import Delaunay


def create_restart_file(refined_ugrid, original_restart_analysis):
    """
    Create new restart file with refined grid dimensions and all variables (initialized to zero).
    
    Parameters
    ----------
    refined_ugrid : xugrid.UgridDataset
        Refined grid dataset
    original_restart_analysis : dict
        Analysis results from original restart file
        
    Returns
    -------
    xarray.Dataset
        New restart file with correct dimensions and all variables
    """
    original_ds = original_restart_analysis['dataset']
    
    # Get refined grid dimensions
    n_refined_faces = refined_ugrid.sizes['mesh2d_nFaces']
    n_refined_edges = refined_ugrid.sizes['mesh2d_nEdges']
    
    # Create coordinate variables from refined grid
    coords = {
        'time': original_ds.coords['time'],
        'FlowElem_xcc': xr.DataArray(
            refined_ugrid.mesh2d_face_x.values,
            dims=['nFlowElem'],
            attrs={
                'units': 'degrees_east',
                'standard_name': 'longitude',
                'long_name': 'x-coordinate of flow element circumcenter'
            }
        ),
        'FlowElem_ycc': xr.DataArray(
            refined_ugrid.mesh2d_face_y.values,
            dims=['nFlowElem'],
            attrs={
                'units': 'degrees_north', 
                'standard_name': 'latitude',
                'long_name': 'y-coordinate of flow element circumcenter'
            }
        ),
        'FlowLink_xu': xr.DataArray(
            refined_ugrid.mesh2d_edge_x.values,
            dims=['nFlowLink'],
            attrs={
                'units': 'degrees_east',
                'standard_name': 'longitude', 
                'long_name': 'x-coordinate of flow link center (velocity point)'
            }
        ),
        'FlowLink_yu': xr.DataArray(
            refined_ugrid.mesh2d_edge_y.values,
            dims=['nFlowLink'],
            attrs={
                'units': 'degrees_north',
                'standard_name': 'latitude',
                'long_name': 'y-coordinate of flow link center (velocity point)'
            }
        ),
        'FlowElem_xbnd': xr.DataArray(
            np.array([]),
            dims=['nFlowElemBnd'],
            attrs={'long_name': 'longitude for boundary points'}
        ),
        'FlowElem_ybnd': xr.DataArray(
            np.array([]),
            dims=['nFlowElemBnd'],
            attrs={'long_name': 'latitude for boundary points'}
        )
    }
    
    # Create dataset with coordinates
    new_restart_ds = xr.Dataset(coords=coords)
    
    # Create all data variables with correct dimensions (initialized to zero)
    variable_info = original_restart_analysis['variable_info']
    
    for var_name, var_info in variable_info.items():
        # Map old dimensions to new dimensions
        new_dims = []
        new_shape = []
        
        for dim in var_info['dims']:
            if dim in ['nFlowElem', 'nNetElem']:
                new_dims.append('nFlowElem')
                new_shape.append(n_refined_faces)
            elif dim in ['nFlowLink', 'nNetLink']:
                new_dims.append('nFlowLink')
                new_shape.append(n_refined_edges)
            elif dim == 'nFlowElemBnd':
                new_dims.append('nFlowElemBnd')
                new_shape.append(0)  # Empty for now
            else:
                # Keep original dimension (time, laydim, wdim, etc.)
                new_dims.append(dim)
                new_shape.append(original_ds.sizes[dim])
        
        # Create zero-filled array with correct shape and dtype
        dtype = np.int32 if var_info['dtype'].startswith('int') else np.float64
        fill_value = 0 if var_info['dtype'].startswith('int') else 0.0
        zero_data = np.full(new_shape, fill_value, dtype=dtype)
        
        # Create DataArray with attributes
        new_restart_ds[var_name] = xr.DataArray(
            zero_data,
            dims=new_dims,
            attrs=var_info['attributes']
        )
    
    # Copy global attributes from original
    new_restart_ds.attrs = original_ds.attrs.copy()
    new_restart_ds.attrs['history'] = f"Created from refined grid on {datetime.now().isoformat()}"
    
    return new_restart_ds


def regrid_restart_data(original_ds, new_restart_ds, refined_ugrid, ugrid_original):
    """
    Regrid data from original restart to refined grid using triangulation interpolation.
    
    Parameters
    ----------
    original_ds : xarray.Dataset
        Original restart dataset
    new_restart_ds : xarray.Dataset
        New restart dataset (modified in place)
    refined_ugrid : xugrid.UgridDataset
        Refined grid dataset
    ugrid_original : xugrid.UgridDataset
        Original grid dataset
    """
    print("🔄 Starting restart data regridding...")
    
    def categorize_variables(dataset):
        """Categorize variables by location and dimensions."""
        elem_1d, elem_2d, link_1d, link_2d = [], [], [], []
        for var_name, var in dataset.data_vars.items():
            dims = var.dims
            if 'time' not in dims:
                continue
            if 'nFlowElem' in dims:
                if len(dims) == 2:
                    elem_1d.append(var_name)
                elif len(dims) == 3:
                    elem_2d.append(var_name)
            elif 'nFlowLink' in dims:
                if len(dims) == 2:
                    link_1d.append(var_name)
                elif len(dims) == 3:
                    link_2d.append(var_name)
        return elem_1d, elem_2d, link_1d, link_2d
    
    def interp_weights(xy, uv, d=2):
        """Compute interpolation weights using triangulation."""
        tri = Delaunay(xy)
        simplex = tri.find_simplex(uv)
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = uv - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    
    def interpolate_values(values, vertices, weights, fill_value=0.0):
        """Interpolate values using precomputed weights."""
        interpolated = np.einsum('nj,nj->n', np.take(values, vertices), weights)
        interpolated[np.any(weights < 0, axis=1)] = fill_value
        return interpolated
    
    # Categorize variables
    elem_1d, elem_2d, link_1d, link_2d = categorize_variables(original_ds)
    print(f"   Elements: {len(elem_1d)} 1D + {len(elem_2d)} 2D variables")
    print(f"   Links: {len(link_1d)} 1D + {len(link_2d)} 2D variables")
    
    # Get coordinates
    original_elem_coords = np.column_stack([ugrid_original.mesh2d_face_x.values, ugrid_original.mesh2d_face_y.values])
    original_link_coords = np.column_stack([ugrid_original.mesh2d_edge_x.values, ugrid_original.mesh2d_edge_y.values])
    refined_elem_coords = np.column_stack([refined_ugrid.mesh2d_face_x.values, refined_ugrid.mesh2d_face_y.values])
    refined_link_coords = np.column_stack([refined_ugrid.mesh2d_edge_x.values, refined_ugrid.mesh2d_edge_y.values])
    
    # Compute interpolation weights for elements
    print("🔍 Computing element interpolation weights...")
    elem_vertices, elem_weights = interp_weights(original_elem_coords, refined_elem_coords)
    elem_inside = np.all(elem_weights > -1e-6, axis=1)
    elem_inside_count = np.sum(elem_inside)
    print(f"📍 Element mapping: {elem_inside_count:,} inside, {len(refined_elem_coords) - elem_inside_count:,} outside")
    
    # Compute interpolation weights for links
    print("🔍 Computing link interpolation weights...")
    link_vertices, link_weights = interp_weights(original_link_coords, refined_link_coords)
    link_inside = np.all(link_weights > -1e-6, axis=1)
    link_inside_count = np.sum(link_inside)
    print(f"🔗 Link mapping: {link_inside_count:,} inside, {len(refined_link_coords) - link_inside_count:,} outside")
    
    # Process element variables
    print("⚙️  Processing element variables...")
    total_elem_vars = len(elem_1d) + len(elem_2d)
    for i, var_name in enumerate(elem_1d + elem_2d, 1):
        if var_name in new_restart_ds:
            print(f"   Processing {var_name} ({i}/{total_elem_vars})")
            
            original_data = original_ds[var_name].values
            new_data = new_restart_ds[var_name].values
            
            if len(original_data.shape) == 2:  # 1D variable
                interpolated = interpolate_values(original_data[0], elem_vertices, elem_weights, fill_value=0.0)
                new_data[0] = interpolated
            else:  # 2D variable with layers
                for layer in range(original_data.shape[2]):
                    interpolated = interpolate_values(original_data[0, :, layer], elem_vertices, elem_weights, fill_value=0.0)
                    new_data[0, :, layer] = interpolated
    
    # Process link variables
    print("⚙️  Processing link variables...")
    total_link_vars = len(link_1d) + len(link_2d)
    for i, var_name in enumerate(link_1d + link_2d, 1):
        if var_name in new_restart_ds:
            print(f"   Processing {var_name} ({i}/{total_link_vars})")
            
            original_data = original_ds[var_name].values
            new_data = new_restart_ds[var_name].values
            
            if len(original_data.shape) == 2:  # 1D variable
                interpolated = interpolate_values(original_data[0], link_vertices, link_weights, fill_value=0.0)
                new_data[0] = interpolated
            else:  # 2D variable with layers
                for layer in range(original_data.shape[2]):
                    interpolated = interpolate_values(original_data[0, :, layer], link_vertices, link_weights, fill_value=0.0)
                    new_data[0, :, layer] = interpolated
    
    print("✅ Restart data regridding complete!")


def compare_restart_files(original_analysis, new_restart_ds):
    """
    Compare original and new restart files in table format.
    
    Parameters
    ----------
    original_analysis : dict
        Analysis results from original restart file
    new_restart_ds : xarray.Dataset
        New restart dataset
    """
    original_ds = original_analysis['dataset']
    
    print("📊 RESTART FILE COMPARISON")
    print("=" * 120)
    
    # Compare dimensions
    print("📏 DIMENSIONS:")
    print(f"{'Dimension Name':<25} {'Original Size':<15} {'New Size':<15} {'Change':<20}")
    print("-" * 80)
    
    for dim_name in original_ds.sizes.keys():
        orig_size = original_ds.sizes[dim_name]
        new_size = new_restart_ds.sizes.get(dim_name, 0)
        
        if new_size == 0:
            change = "Missing"
        elif new_size == orig_size:
            change = "Same"
        else:
            ratio = new_size / orig_size if orig_size > 0 else 0
            change = f"{ratio:.1f}x"
        
        print(f"{dim_name:<25} {orig_size:<15,} {new_size:<15,} {change:<20}")
    
    print()
    
    # Summary statistics
    coord_original_mb = sum(coord.size * coord.dtype.itemsize / (1024**2) for coord in original_ds.coords.values())
    coord_new_mb = sum(coord.size * coord.dtype.itemsize / (1024**2) for coord in new_restart_ds.coords.values())
    
    total_original_mb = sum(original_analysis['variable_info'][var]['size_mb'] for var in original_analysis['variable_info'])
    total_new_mb = sum(var.size * var.dtype.itemsize / (1024**2) for var in new_restart_ds.data_vars.values())
    
    total_orig_all = coord_original_mb + total_original_mb
    total_new_all = coord_new_mb + total_new_mb
    size_ratio = total_new_all / total_orig_all if total_orig_all > 0 else 0
    
    print(f"💾 SUMMARY:")
    print(f"   Original file size: {original_analysis['file_size_gb']:.2f} GB")
    print(f"   New file size (estimated): {total_new_all/1024:.2f} GB")
    print(f"   Size increase factor: {size_ratio:.1f}x")
    print("=" * 120)