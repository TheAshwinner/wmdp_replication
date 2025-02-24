import argparse
import yaml
from typing import Dict, Any
import json
import numpy as np
import torch

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        Dict[str, Any]: Dictionary containing the configuration parameters
        
    Example:
        >>> config = load_config("yaml_files/finetuning/gpt2.yaml")
        >>> print(config["model_name"])
        'openai-community/gpt2'
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
            
    args = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                setattr(args, sub_key, sub_value)
        else:
            setattr(args, key, value)
    
    return args

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for handling special PyTorch and NumPy types.

    This encoder handles:
    - NumPy integers and floats
    - NumPy arrays
    - PyTorch tensors
    - PyTorch dtypes
    - Callable objects
    """

    def default(self, obj: Any) -> Any:
        """
        Convert special types to JSON-serializable formats.

        Args:
            obj: Object to be serialized

        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, torch.dtype):
            return str(obj)
        if callable(obj):
            return obj.__name__
        return super(CustomJSONEncoder, self).default(obj)