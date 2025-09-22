"""
Template for modifying variables in D-Flow FM restart files.
Copy this file and modify for your specific variable.
"""

import numpy as np


def modify_your_variable(datasets, your_parameter):
    """
    Template function - copy and modify this for your variable.
    
    Parameters
    ----------
    datasets : list
        List of (partition_number, dataset) tuples
    your_parameter : float
        Your modification parameter
        
    Returns
    -------
    list
        List of (partition_number, modified_dataset) tuples
    """
    modified_datasets = []
    
    for partition_num, dataset in datasets:
        # Modify your variable - replace 'your_variable_name' and modification logic
        var_data = dataset['your_variable_name'].values.copy()
        var_data *= your_parameter  # Example: multiply by factor
        
        # Update dataset
        dataset_modified = dataset.copy()
        dataset_modified['your_variable_name'] = (dataset['your_variable_name'].dims, var_data)
        modified_datasets.append((partition_num, dataset_modified))
    
    return modified_datasets