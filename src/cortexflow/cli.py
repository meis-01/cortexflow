#!/usr/bin/env python
"""
CLI utility to manage CortexFlow mode switching between pilot and final datasets.
"""
import argparse
import json
from pathlib import Path
from cortexflow.config_utils import load_config, get_config_path, get_data_paths


def cmd_status():
    """Show current mode configuration."""
    try:
        # Check params.yaml for current mode
        import yaml
        params_path = Path("params.yaml")
        if params_path.exists():
            with open(params_path) as f:
                params = yaml.safe_load(f)
            current_mode = params.get("mode", "unknown")
        else:
            current_mode = "unknown (params.yaml not found)"
        
        print(f"Current mode: {current_mode}")
        
        # Show available configs
        config_dir = Path("configs")
        configs = list(config_dir.glob("*.json"))
        print(f"\nAvailable configs ({len(configs)}):")
        for cfg in sorted(configs):
            print(f"  - {cfg.stem}")
            config = json.loads(cfg.read_text())
            if "dataset" in config:
                print(f"    Data dir: {config['dataset'].get('text_trs', '').rsplit('/', 1)[0]}")
        
    except Exception as e:
        print(f"Error showing status: {e}")


def cmd_switch(mode: str):
    """Switch to a specific mode (pilot or final)."""
    valid_modes = ["pilot", "final"]
    if mode not in valid_modes:
        print(f"Error: mode must be one of {valid_modes}")
        return False
    
    try:
        # Verify config exists
        config = load_config(mode)
        print(f"✓ Config found: {get_config_path(mode)}")
        
        # Update params.yaml
        import yaml
        params_path = Path("params.yaml")
        if params_path.exists():
            with open(params_path) as f:
                params = yaml.safe_load(f) or {}
        else:
            params = {}
        
        params["mode"] = mode
        with open(params_path, "w") as f:
            yaml.dump(params, f, default_flow_style=False)
        print(f"✓ Updated params.yaml to mode: {mode}")
        
        # Show data paths
        paths = get_data_paths(mode)
        print(f"\nData paths for {mode} mode:")
        for key, path in sorted(paths.items()):
            print(f"  {key}: {path}")
        
        print(f"\n✓ Switched to {mode} mode")
        print(f"Run experiments with: python src/run_paper.py --mode {mode}")
        return True
        
    except Exception as e:
        print(f"Error switching mode: {e}")
        return False


def cmd_show(mode: str):
    """Show full config for a mode."""
    try:
        config = load_config(mode)
        print(f"Config for {mode} mode ({get_config_path(mode)}):")
        print(json.dumps(config, indent=2))
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CortexFlow mode manager: Switch between pilot and final datasets"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    subparsers.add_parser("status", help="Show current mode and available configs")
    
    switch_parser = subparsers.add_parser("switch", help="Switch to a mode")
    switch_parser.add_argument("mode", choices=["pilot", "final"])
    
    show_parser = subparsers.add_parser("show", help="Show full config for a mode")
    show_parser.add_argument("mode", choices=["pilot", "final"])
    
    args = parser.parse_args()
    
    if args.command == "status":
        cmd_status()
    elif args.command == "switch":
        cmd_switch(args.mode)
    elif args.command == "show":
        cmd_show(args.mode)
    else:
        parser.print_help()
