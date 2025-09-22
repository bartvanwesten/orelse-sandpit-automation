"""
Utility functions for DFM model partitioning and restart file generation.
"""

import os
import shutil
import subprocess
import re
from pathlib import Path
import glob
import xarray as xr


def create_temporary_mdu(model_dir, original_mdu, refined_grid_nc, temp_suffix="_temp"):
    """
    Create temporary MDU file with modified parameters for partitioning test.
    
    Parameters
    ----------
    model_dir : Path
        Model directory path
    original_mdu : Path
        Path to original MDU file
    refined_grid_nc : Path
        Path to refined grid NetCDF file
    temp_suffix : str
        Suffix for temporary MDU file
        
    Returns
    -------
    Path
        Path to created temporary MDU file
    """
    temp_mdu = model_dir / f"{original_mdu.stem}{temp_suffix}{original_mdu.suffix}"
    shutil.copy(original_mdu, temp_mdu)
    
    # Copy refined grid to model directory
    refined_grid_local = model_dir / refined_grid_nc.name
    if not refined_grid_local.exists():
        shutil.copy(refined_grid_nc, refined_grid_local)
    
    # Read and modify MDU file
    with open(temp_mdu, 'r') as f:
        mdu_content = f.read()
    
    # Define modifications
    modifications = {
        'NetFile': refined_grid_nc.name,
        'Stopdatetime': '20180101000002',
        'MapInterval': '600.',
        'RstInterval': '1. 1. 2.',
        'HisInterval': '1.',
        'DtUser': '1.',
        'DtMax': '1.',
    }
    
    # Apply modifications
    mdu_lines = mdu_content.split('\n')
    for i, line in enumerate(mdu_lines):
        for key, value in modifications.items():
            if line.strip().startswith(key) and '=' in line:
                mdu_lines[i] = f"{key:<40} = {value:<60} # Modified for partitioning test"
    
    # Write modified MDU
    with open(temp_mdu, 'w') as f:
        f.write('\n'.join(mdu_lines))
    
    return temp_mdu


def create_temporary_dimr_config(model_dir, dimr_config, temp_mdu, n_partitions, temp_suffix="_temp"):
    """
    Create temporary DIMR config file with updated partition settings.
    
    Parameters
    ----------
    model_dir : Path
        Model directory path
    dimr_config : Path
        Path to original DIMR config file
    temp_mdu : Path
        Path to temporary MDU file
    n_partitions : int
        Number of partitions
    temp_suffix : str
        Suffix for temporary config file
        
    Returns
    -------
    Path
        Path to created temporary DIMR config file
    """
    # Read original XML
    with open(dimr_config, 'r', encoding='iso-8859-1') as f:
        xml_content = f.read()
    
    # Update process element
    process_pattern = r'(<process>)[^<]*(</process>)'
    process_list = ' '.join(str(i) for i in range(n_partitions))
    new_process = f'\\g<1>{process_list}\\g<2>'
    xml_content = re.sub(process_pattern, new_process, xml_content)
    
    # Update inputFile reference - derive original MDU name from temp MDU
    original_mdu_name = temp_mdu.name.replace('_temp', '')
    xml_content = xml_content.replace(original_mdu_name, temp_mdu.name)
    
    # Save modified XML
    temp_dimr_config = model_dir / f"{dimr_config.stem}{temp_suffix}{dimr_config.suffix}"
    with open(temp_dimr_config, 'w', encoding='iso-8859-1') as f:
        f.write(xml_content)
    
    return temp_dimr_config


def partition_grid(dflowfm_exe, temp_mdu, n_partitions):
    """
    Partition the refined grid using DFlow FM.
    
    Parameters
    ----------
    dflowfm_exe : Path
        Path to DFlow FM executable
    temp_mdu : Path
        Path to temporary MDU file
    n_partitions : int
        Number of partitions
        
    Returns
    -------
    bool
        True if partitioning succeeded, False otherwise
    """
    partition_cmd = [
        str(dflowfm_exe),
        f'--partition:ndomains={n_partitions}:icgsolver=6',
        str(temp_mdu.name)
    ]
    
    print(f"Partition command: {' '.join(partition_cmd)}")
    
    result = subprocess.run(partition_cmd, capture_output=True, text=True, shell=True)
    
    print("Partitioning output:")
    print(result.stdout)
    if result.stderr:
        print("Partitioning errors:")
        print(result.stderr)
    
    if result.returncode == 0:
        # Check if partition files were created
        partition_files = list(Path('.').glob(f"{temp_mdu.stem}_0*{temp_mdu.suffix}"))
        success = len(partition_files) == n_partitions
        if success:
            print(f"Partitioning completed successfully")
            print(f"Created {len(partition_files)} partition MDU files")
        else:
            print(f"Partitioning failed: expected {n_partitions} files, got {len(partition_files)}")
        return success
    else:
        print(f"Partitioning failed with return code {result.returncode}")
        return False


def run_parallel_model(dimr_parallel_exe, temp_dimr_config, n_partitions):
    """
    Run parallel model to generate restart files.
    
    Parameters
    ----------
    dimr_parallel_exe : Path
        Path to DIMR parallel executable
    temp_dimr_config : Path
        Path to temporary DIMR config file
    n_partitions : int
        Number of partitions
        
    Returns
    -------
    bool
        True if model run succeeded, False otherwise
    """
    parallel_cmd = [
        str(dimr_parallel_exe),
        str(n_partitions),
        str(temp_dimr_config.name)
    ]
    
    print(f"Parallel run command: {' '.join(parallel_cmd)}")
    
    result = subprocess.run(parallel_cmd, capture_output=True, text=True, shell=True)
    
    print("Model run output:")
    print(result.stdout)
    if result.stderr:
        print("Model run errors:")
        print(result.stderr)
    
    if result.returncode == 0:
        # Check for restart files
        rst_files = list(Path('.').glob("*_rst.nc"))
        success = len(rst_files) > 0
        if success:
            print(f"Model run completed successfully")
            print(f"Generated {len(rst_files)} restart files")
        else:
            print("Model run succeeded but no restart files found")
        return success
    else:
        print(f"Model run failed with return code {result.returncode}")
        return False


def setup_partitioned_model(model_dir, original_mdu, dimr_config, refined_grid_nc, 
                           dflowfm_exe, dimr_parallel_exe, n_partitions):
    """
    Complete workflow to setup partitioned model with refined grid.
    
    Parameters
    ----------
    model_dir : Path
        Model directory path
    original_mdu : Path
        Path to original MDU file
    dimr_config : Path
        Path to DIMR config file
    refined_grid_nc : Path
        Path to refined grid NetCDF file
    dflowfm_exe : Path
        Path to DFlow FM executable
    dimr_parallel_exe : Path
        Path to DIMR parallel executable
    n_partitions : int
        Number of partitions
        
    Returns
    -------
    dict
        Dictionary with results and file paths
    """
    # Store original working directory and resolve absolute paths first
    original_cwd = Path.cwd()
    model_dir = Path(model_dir).resolve()
    original_mdu = Path(original_mdu).resolve()
    dimr_config = Path(dimr_config).resolve()
    refined_grid_nc = Path(refined_grid_nc).resolve()
    
    # Create temporary files first while in original directory
    temp_mdu = create_temporary_mdu(model_dir, original_mdu, refined_grid_nc)
    temp_dimr_config = create_temporary_dimr_config(model_dir, dimr_config, temp_mdu, n_partitions)
    
    # Change to model directory for partitioning
    os.chdir(model_dir)
    
    # Partition grid
    partition_success = partition_grid(dflowfm_exe, temp_mdu, n_partitions)
    
    # Run parallel model
    model_success = False
    if partition_success:
        model_success = run_parallel_model(dimr_parallel_exe, temp_dimr_config, n_partitions)
    
    # Get created files
    partition_files = list(Path('.').glob(f"{temp_mdu.stem}_0*{temp_mdu.suffix}"))
    rst_files = list(Path('.').glob("*_rst.nc"))
    
    # Restore working directory
    os.chdir(original_cwd)
    
    return {
        'partition_success': partition_success,
        'model_success': model_success,
        'temp_mdu': temp_mdu,
        'temp_dimr_config': temp_dimr_config,
        'partition_files': partition_files,
        'restart_files': rst_files
    }

def load_restart_files(model_dir):
    """
    Load all partitioned restart files from model directory.
    
    Parameters
    ----------
    model_dir : Path
        Model directory containing restart files
        
    Returns
    -------
    list
        List of (partition_number, dataset) tuples sorted by partition number
    """
    restart_files = glob.glob(str(model_dir / "*_rst.nc"))
    datasets = []
    
    for file_path in restart_files:
        print(f"Loading restart file: {file_path}")
        filename = Path(file_path).stem
        partition_num = None
        for part in filename.split('_'):
            if part.isdigit() and len(part) == 4:
                partition_num = int(part)
                break
        
        if partition_num is not None:
            ds = xr.open_dataset(file_path, decode_timedelta=False)
            datasets.append((partition_num, ds))
    
    return sorted(datasets, key=lambda x: x[0])


def save_restart_files(datasets, output_dir, suffix="_modified"):
    """
    Save modified restart datasets to output directory.
    
    Parameters
    ----------
    datasets : list
        List of (partition_number, dataset) tuples
    output_dir : Path
        Output directory for modified restart files
    suffix : str
        Suffix to add to output filenames
        
    Returns
    -------
    list
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    for partition_num, dataset in datasets:
        filename = f"restart_{partition_num:04d}{suffix}.nc"
        output_path = output_dir / filename
        dataset.to_netcdf(output_path)
        saved_files.append(output_path)
        dataset.close()
    
    return saved_files