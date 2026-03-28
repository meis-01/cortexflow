"""
Utilities for mode-based configuration management.
Supports switching between 'pilot' (synthetic) and 'final' (real data) modes.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(mode: str = "pilot", config_dir: str = "configs") -> Dict[str, Any]:
    """
    Load a mode-specific configuration file.
    
    Args:
        mode: Either 'pilot' (synthetic data) or 'final' (real data)
        config_dir: Directory where config files are stored
        
    Returns:
        Dictionary with all configuration parameters
        
    Raises:
        FileNotFoundError: If config file for mode does not exist
        ValueError: If mode is not recognized
    """
    valid_modes = ["pilot", "final"]
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got {mode}")
    
    config_path = Path(config_dir) / f"{mode}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    return config


def get_config_path(mode: str = "pilot", config_dir: str = "configs") -> Path:
    """Get the path to the config file for a given mode."""
    return Path(config_dir) / f"{mode}.json"


def save_config(config: Dict[str, Any], mode: str, config_dir: str = "configs") -> Path:
    """Save a config dictionary to a mode-specific file."""
    config_path = Path(config_dir) / f"{mode}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


def get_data_paths(mode: str = "pilot", config_dir: str = "configs") -> Dict[str, str]:
    """
    Extract all data paths from a mode config.
    Useful for understanding what artifacts will be created.
    """
    config = load_config(mode, config_dir)
    paths = {}
    
    # Flatten nested dicts to collect all paths
    def extract_paths(d, prefix=""):
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                extract_paths(value, full_key)
            elif isinstance(value, str) and ("/" in value or value.endswith(".npy") or value.endswith(".txt")):
                paths[full_key] = value
    
    extract_paths(config)
    return paths
