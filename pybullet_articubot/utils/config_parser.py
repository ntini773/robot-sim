"""
Configuration parser utilities.
Handles YAML config parsing and string-to-array conversions.
"""

import yaml
import numpy as np
import os
from typing import Dict, Any, List, Tuple

def parse_center(center_str: str) -> List[float]:
    """
    Parse a string representation of center coordinates.
    Example: "(0.5, 0.0, 0.2)" -> [0.5, 0.0, 0.2]
    
    Args:
        center_str: String representation
        
    Returns:
        List of floats
    """
    if isinstance(center_str, list) or isinstance(center_str, tuple):
        return list(center_str)
        
    s = center_str.replace('(', '').replace(')', '')
    parts = s.split(',')
    return [float(p.strip()) for p in parts]

def parse_config(config_path: str) -> Tuple[List[str], List[float], List[List[float]], List[List[float]], List[str], List[str], List[bool], bool]:
    """
    Parse the simulation configuration YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Tuple of lists containing object properties:
        (urdf_paths, sizes, positions, orientations, names, types, on_table_flags, use_table_flag)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    urdf_paths = []
    sizes = []
    positions = []
    orientations = []
    names = []
    types = []
    on_tables = []
    use_table = False
    
    for obj in config:
        if 'use_table' in obj:
            use_table = obj['use_table']
        
        # Skip meta entries
        if 'type' not in obj:
            continue
            
        # Parse based on type
        if obj['type'] in ['urdf', 'mesh']:
            # Determine path
            if 'path' in obj:
                path = obj['path']
            elif 'reward_asset_path' in obj:
                # Handle PartNet-Mobility ID or similar
                # For this simplified setup, we assume direct paths or handle specific IDs
                path = obj['reward_asset_path']
            else:
                continue # Skip if no path
            
            urdf_paths.append(path)
            types.append(obj['type'])
            
            # Parse properties
            sizes.append(obj.get('size', 1.0))
            
            center = obj.get('center', [0, 0, 0])
            positions.append(parse_center(center))
            
            orient = obj.get('orientation', [0, 0, 0])
             # If orientation is Euler (size 3) or Quaternion (size 4), handle accordingly in Sim
            orientations.append(parse_center(orient))
            
            names.append(obj.get('name', 'unknown'))
            on_tables.append(obj.get('on_table', False))
            
    return urdf_paths, sizes, positions, orientations, names, types, on_tables, use_table
