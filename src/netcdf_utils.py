"""
NetCDF utilities for mesh grid export and management.
"""

import os
import numpy as np
import xarray as xr
import xugrid as xu
import dfm_tools as dfmt

def generate_versioned_filename(original_file, output_dir, suffix="_ref", extension="_net.nc"):
    """
    Generate a versioned filename to avoid overwriting existing files.
    
    Parameters:
    -----------
    original_file : str
        Path to the original file
    output_dir : str
        Directory where the new file will be saved
    suffix : str, default "_ref"
        Suffix to add before version number
    extension : str, default "_net.nc"
        File extension
        
    Returns:
    --------
    str : The versioned filename
    str : Full path to the output file
    """
    original_name = os.path.basename(original_file)
    base_name = original_name.replace('_net.nc', '')
    
    version = 1
    while True:
        versioned_filename = f"{base_name}{suffix}_v{version:02d}{extension}"
        output_path = os.path.join(output_dir, versioned_filename)
        if not os.path.exists(output_path):
            break
        version += 1
    
    return versioned_filename, output_path


def create_face_node_connectivity(mesh2d):
    """
    Create UGRID-compatible face_node_connectivity array from MeshKernel mesh data.
    
    Parameters:
    -----------
    mesh2d : MeshKernel mesh object
        Mesh data from MeshKernel
        
    Returns:
    --------
    np.ndarray : Face-node connectivity array with fill values for unused nodes
    """
    max_nodes_per_face = mesh2d.nodes_per_face.max()
    n_faces = len(mesh2d.face_x)
    face_nodes_ugrid = np.full((n_faces, max_nodes_per_face), -1, dtype=np.int32)
    
    start_idx = 0
    for i, nodes_in_face in enumerate(mesh2d.nodes_per_face):
        end_idx = start_idx + nodes_in_face
        face_nodes_ugrid[i, :nodes_in_face] = mesh2d.face_nodes[start_idx:end_idx]
        start_idx = end_idx
    
    return face_nodes_ugrid


def calculate_edge_coordinates(mesh2d):
    """
    Calculate edge center coordinates from node coordinates.
    
    Parameters:
    -----------
    mesh2d : MeshKernel mesh object
        Mesh data from MeshKernel
        
    Returns:
    --------
    tuple : (edge_x, edge_y, edge_nodes_2d) arrays
    """
    edge_nodes_2d = mesh2d.edge_nodes.reshape(-1, 2)
    edge_x = (mesh2d.node_x[edge_nodes_2d[:, 0]] + mesh2d.node_x[edge_nodes_2d[:, 1]]) / 2
    edge_y = (mesh2d.node_y[edge_nodes_2d[:, 0]] + mesh2d.node_y[edge_nodes_2d[:, 1]]) / 2
    
    return edge_x, edge_y, edge_nodes_2d


def create_ugrid_dataset(mesh2d, crs='EPSG:4326'):
    """
    Create a complete UGRID-compatible xarray Dataset from MeshKernel mesh data.
    
    Parameters:
    -----------
    mesh2d : MeshKernel mesh object
        Mesh data from MeshKernel
    crs : str, default 'EPSG:4326'
        Coordinate reference system
        
    Returns:
    --------
    xr.Dataset : UGRID-compatible dataset
    xu.Ugrid2d : UGRID grid object
    dict : Mesh statistics (nodes, faces, edges counts)
    """
    # Get mesh dimensions
    n_nodes = len(mesh2d.node_x)
    n_faces = len(mesh2d.face_x)
    n_edges = len(mesh2d.edge_nodes) // 2
    
    # Create connectivity and coordinate arrays
    face_nodes_ugrid = create_face_node_connectivity(mesh2d)
    edge_x, edge_y, edge_nodes_2d = calculate_edge_coordinates(mesh2d)
    max_nodes_per_face = face_nodes_ugrid.shape[1]
    
    # Create UGRID grid object with spherical coordinates (projected=False)
    grid = xu.Ugrid2d(
        node_x=mesh2d.node_x, 
        node_y=mesh2d.node_y, 
        face_node_connectivity=face_nodes_ugrid, 
        fill_value=-1, 
        start_index=0,
        projected=False,  # FALSE for spherical/geographic coordinates
        crs=crs
    )
    
    # Create xarray Dataset with UGRID conventions
    ds = xr.Dataset({
        'mesh2d_node_x': (('mesh2d_nNodes',), mesh2d.node_x, 
                         {'standard_name': 'longitude', 'units': 'degrees_east'}),
        'mesh2d_node_y': (('mesh2d_nNodes',), mesh2d.node_y, 
                         {'standard_name': 'latitude', 'units': 'degrees_north'}),
        'mesh2d_face_x': (('mesh2d_nFaces',), mesh2d.face_x, 
                         {'standard_name': 'longitude', 'units': 'degrees_east'}),
        'mesh2d_face_y': (('mesh2d_nFaces',), mesh2d.face_y, 
                         {'standard_name': 'latitude', 'units': 'degrees_north'}),
        'mesh2d_edge_x': (('mesh2d_nEdges',), edge_x, 
                         {'standard_name': 'longitude', 'units': 'degrees_east'}),
        'mesh2d_edge_y': (('mesh2d_nEdges',), edge_y, 
                         {'standard_name': 'latitude', 'units': 'degrees_north'}),
        'mesh2d_face_nodes': (('mesh2d_nFaces', 'mesh2d_nMax_face_nodes'), face_nodes_ugrid, 
                             {'cf_role': 'face_node_connectivity', 'start_index': 0, '_FillValue': -1}),
        'mesh2d_edge_nodes': (('mesh2d_nEdges', 'Two'), edge_nodes_2d, 
                             {'cf_role': 'edge_node_connectivity', 'start_index': 0}),
        'mesh2d': ((), 0, {
            'cf_role': 'mesh_topology', 
            'topology_dimension': 2,
            'node_coordinates': 'mesh2d_node_x mesh2d_node_y', 
            'node_dimension': 'mesh2d_nNodes',
            'face_coordinates': 'mesh2d_face_x mesh2d_face_y', 
            'face_dimension': 'mesh2d_nFaces',
            'face_node_connectivity': 'mesh2d_face_nodes', 
            'max_face_nodes_dimension': 'mesh2d_nMax_face_nodes',
            'edge_coordinates': 'mesh2d_edge_x mesh2d_edge_y', 
            'edge_dimension': 'mesh2d_nEdges',
            'edge_node_connectivity': 'mesh2d_edge_nodes'
        }),
        # Updated WGS84 coordinate reference system for spherical coordinates
        'wgs84': ((), 0, {
            'name': 'wgs 84', #WGS84
            'epsg': np.int32(4326), 
            'grid_mapping_name': 'latitude_longitude',
            'EPSG_code': 'EPSG:4326'
        })
    }, coords={
        'mesh2d_nNodes': np.arange(n_nodes),
        'mesh2d_nFaces': np.arange(n_faces),
        'mesh2d_nEdges': np.arange(n_edges),
        'mesh2d_nMax_face_nodes': np.arange(max_nodes_per_face),
        'Two': np.arange(2)
    })
    
    # Mesh statistics
    stats = {
        'nodes': n_nodes,
        'faces': n_faces,
        'edges': n_edges
    }
    
    return ds, grid, stats

def export_refined_grid(mk_object, original_nc_file, output_dir, suffix="_ref", verbose=True):
    """
    Export a refined mesh grid to NetCDF with proper UGRID naming and compatibility.
    
    Parameters:
    -----------
    mk_object : MeshKernel object
        MeshKernel instance containing the refined mesh
    original_nc_file : str
        Path to the original NetCDF file (used for naming)
    output_dir : str
        Directory where the refined grid will be saved
    suffix : str, default "_ref"
        Suffix for the refined grid filename
    verbose : bool, default True
        Whether to print progress messages
        
    Returns:
    --------
    str : Path to the saved NetCDF file
    dict : Mesh statistics
    xu.UgridDataset : The complete UGRID dataset object
    """
    if verbose:
        print("💾 Saving refined grid...")
    
    # Generate versioned filename
    filename, output_path = generate_versioned_filename(
        original_nc_file, output_dir, suffix
    )
    

    ugrid_complete = dfmt.meshkernel_to_UgridDataset(mk=mk_object, crs='EPSG:4326')

    # # Get mesh data from MeshKernel
    # mesh2d = mk_object.mesh2d_get()
    
    # # Create UGRID dataset with spherical coordinates
    # ds, grid, stats = create_ugrid_dataset(mesh2d, crs='EPSG:4326')
    
    # # Create UgridDataset and save
    # ugrid_complete = xu.UgridDataset(ds, grids=[grid])
    # ugrid_complete.ugrid.set_crs('EPSG:4326')

    ugrid_complete.ugrid.to_netcdf(output_path)


    if verbose:
        print(f"✅ Refined grid saved to: {filename}")
        # print(f"   Nodes: {stats['nodes']:,} | Faces: {stats['faces']:,} | Edges: {stats['edges']:,}")
    
    return output_path, ugrid_complete # stats, ugrid_complete