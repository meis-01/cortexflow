#!/usr/bin/env python
"""
Batch experiment runner for LORO across multiple models/languages/layers.

Usage:
    python run_loro_batch.py --config batch_config.json --output_dir results/
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List


def run_batch(
    experiments: List[Dict],
    output_dir: str,
    data_root: str,
    ds003643_root: str,
) -> None:
    """Run multiple LORO experiments sequentially or in parallel.

    Parameters
    ----------
    experiments : list of dicts
        Each with keys: model, lang, layer (and optional: n_voxels, hrf_model)
    output_dir : str
        Root output directory (results will be in subdirs)
    data_root, ds003643_root : str
        Paths to data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_log = {
        "experiments": [],
        "summary": {},
    }

    for exp_idx, exp_config in enumerate(experiments, 1):
        model = exp_config["model"]
        lang = exp_config["lang"]
        layer = exp_config["layer"]
        n_voxels = exp_config.get("n_voxels", 100000)
        temporal_feature_strategy = exp_config.get(
            "temporal_feature_strategy", "hemodynamic_convolution"
        )
        response_lag_trs = exp_config.get("response_lag_trs", [2, 3, 4, 5])

        exp_name = f"{model}_{lang}_layer{layer}".replace("/", "_")
        exp_output = output_dir / exp_name

        print(f"\n{'='*70}")
        print(f"[{exp_idx}/{len(experiments)}] Running: {exp_name}")
        print(f"{'='*70}")

        cmd = [
            "python",
            "src/run_loro_experiment.py",
            "--model",
            model,
            "--lang",
            lang,
            "--layer",
            str(layer),
            "--data_root",
            data_root,
            "--ds003643_root",
            ds003643_root,
            "--output_dir",
            str(exp_output),
            "--n_voxels",
            str(n_voxels),
            "--temporal_feature_strategy",
            temporal_feature_strategy,
        ]

        if temporal_feature_strategy == "lagged_response_window":
            cmd.extend(["--response_lag_trs", *[str(delay) for delay in response_lag_trs]])

        result = subprocess.run(cmd, cwd=".")
        if result.returncode != 0:
            print(f"❌ Experiment failed: {exp_name}")
            results_log["experiments"].append(
                {
                    "config": exp_config,
                    "status": "failed",
                    "output_dir": str(exp_output),
                }
            )
        else:
            print(f"✅ Experiment succeeded: {exp_name}")
            # Try to load metrics
            metrics_file = exp_output / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                results_log["experiments"].append(
                    {
                        "config": exp_config,
                        "status": "success",
                        "output_dir": str(exp_output),
                        "metrics": metrics,
                    }
                )
                results_log["summary"][exp_name] = {
                    "mean_corr": metrics["mean_corr"],
                    "median_corr": metrics["median_corr"],
                    "n_voxels": metrics["n_voxels"],
                }

    # Save batch log
    log_file = output_dir / "batch_results.json"
    with open(log_file, "w") as f:
        json.dump(results_log, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Batch complete. Summary saved to: {log_file}")
    print(f"{'='*70}\n")

    # Print summary table
    if results_log["summary"]:
        print("\nSummary:")
        for exp_name, summary in results_log["summary"].items():
            print(
                f"  {exp_name:40s}  "
                f"mean_r={summary['mean_corr']:6.4f}  "
                f"med_r={summary['median_corr']:6.4f}  "
                f"n_vox={summary['n_voxels']:6d}"
            )


def main():
    parser = argparse.ArgumentParser(description="Batch LORO experiment runner.")
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        help="Experiment specs as MODEL:LANG:LAYER (e.g., gpt2:en:6 opt-350m:en:6)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="JSON config file with experiment list",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/loro_batch",
        help="Output directory",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to llms_brain_lateralization",
    )
    parser.add_argument(
        "--ds003643_root",
        type=str,
        default=None,
        help="Path to OpenNeuro ds003643",
    )

    args = parser.parse_args()

    # Load experiments from config or args
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        experiments = config.get("experiments", [])
        if args.data_root is None:
            args.data_root = config.get("data_root")
        if args.ds003643_root is None:
            args.ds003643_root = config.get("ds003643_root")
    elif args.experiments:
        experiments = []
        for spec in args.experiments:
            parts = spec.split(":")
            if len(parts) != 3:
                raise ValueError(f"Invalid spec: {spec} (expected MODEL:LANG:LAYER)")
            experiments.append(
                {"model": parts[0], "lang": parts[1], "layer": int(parts[2])}
            )
    else:
        raise ValueError("Provide either --config or --experiments")

    if args.data_root is None:
        args.data_root = str(Path(__file__).parent.parent / "data")
    if args.ds003643_root is None:
        args.ds003643_root = str(
            Path(__file__).parent.parent / "data" / "openneuro" / "ds003643"
        )

    run_batch(
        experiments=experiments,
        output_dir=args.output_dir,
        data_root=args.data_root,
        ds003643_root=args.ds003643_root,
    )

    return 0


if __name__ == "__main__":
    exit(main())
