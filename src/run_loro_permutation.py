#!/usr/bin/env python
"""
Permutation baseline runner for statistical significance testing.

Permutes word-level LLM activations within each run, then refits LORO to assess
how much of the real correlation is above chance. Used to compute empirical p-values.

Usage:
    python run_loro_permutation.py \\
        --model gpt2 \\
        --lang en \\
        --layer 6 \\
        --n_permutations 20 \\
        --real_results_dir results/loro/gpt2_en_layer6 \\
        --output_dir results/permutation/gpt2_en_layer6
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from cortexflow.design import build_regressor_from_activations
from cortexflow.features import (
    extract_word_activations_run,
    load_llm_model,
)
from cortexflow.fmri import load_avg_subject_runs
from cortexflow.modeling import train_ridge_loro
import pandas as pd
import zipfile
from tqdm import tqdm


def _extract_valid_word_events(df_run: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Keep only valid word rows to match the original paper preprocessing."""
    word_list: list[str] = []
    onsets: list[float] = []
    offsets: list[float] = []
    for word, onset, offset in zip(df_run["word"], df_run["onset"], df_run["offset"]):
        if isinstance(word, str) and word != " ":
            word_list.append(word)
            onsets.append(float(onset))
            offsets.append(float(offset))
    return word_list, np.array(onsets), np.array(offsets)


def run_permutation_baseline(
    model_name: str,
    lang: str,
    layer: int,
    data_root: str,
    ds003643_root: str,
    output_dir: str,
    real_results_dir: Optional[str] = None,
    n_permutations: int = 20,
    n_runs: int = 9,
    t_r: float = 2.0,
    hrf_model: str = "glover",
    n_voxels: int = 100000,
    seed: int = 42,
    access_token: Optional[str] = None,
    permutation_mode: str = "shuffle",  # "shuffle", "circular_shift"
) -> Dict:
    """Run permutation baseline for significance testing.

    Parameters
    ----------
    model_name, lang, layer : str, str, int
        Experiment specification (same as run_loro_experiment.py)
    data_root, ds003643_root : str
        Paths to data
    output_dir : str
        Output directory for permutation results
    real_results_dir : str, optional
        Directory containing real results (for comparison)
    n_permutations : int
        Number of permutations (default 20)
    permutation_mode : str
        "shuffle" (default) or "circular_shift"
    seed : int
        Random seed
    access_token : str, optional
        HF token for gated models

    Returns
    -------
    dict with keys:
        perm_corr (n_perm, n_voxels), real_corr (n_voxels), empirical_p, ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    lang_lower = lang.lower()

    # ---- Load data (identical to real run) ----
    print(f"\nLoading real fMRI & LLM for {model_name} ({lang_lower}, layer {layer})...")

    avg_subject_dir = Path(data_root) / f"lpp_{lang_lower}_average_subject"
    fmri_runs = load_avg_subject_runs(
        avg_subject_dir=str(avg_subject_dir),
        n_runs=n_runs,
        trim_trs=10,
    )

    model, tokenizer, n_layers, maxlen, stride = load_llm_model(
        model_name, access_token=access_token
    )

    # Load word annotations & text
    csv_path = (
        Path(ds003643_root)
        / "annotation"
        / lang_lower.upper()
        / f"lpp{lang_lower.upper()}_word_information.csv"
    )
    df_words = pd.read_csv(csv_path)
    df_words = df_words.reset_index(drop=True)

    required_cols = {"word", "onset", "offset", "section"}
    missing = sorted(required_cols.difference(df_words.columns))
    if missing:
        raise ValueError(
            f"Annotation CSV missing required columns: {missing}. "
            f"Found columns: {list(df_words.columns)}"
        )

    # Apply ad-hoc corrections (safely, only if rows exist)
    if lang_lower == "en":
        rows_to_drop = [i for i in [3919, 6775, 6781] if i < len(df_words)]
        if rows_to_drop:
            df_words = df_words.drop(rows_to_drop).reset_index(drop=True)
    elif lang_lower == "fr":
        for idx, val in [(3332, "de"), (3379, "trois"), (3405, "trois"), (4587, "l"),
                         (5325, "la"), (5326, "première"), (5328, "habitée"), (11257, "À"), (12249, "il")]:
            if idx < len(df_words):
                df_words.loc[idx, "word"] = val
        rows_to_drop = [i for i in [338, 1204, 3333] if i < len(df_words)]
        if rows_to_drop:
            df_words = df_words.drop(rows_to_drop).reset_index(drop=True)

    text_zip = zipfile.ZipFile(
        Path(data_root) / f"lpp_{lang_lower}_text.zip", "r"
    )

    # ---- Extract real activations ----
    print("Extracting word-level activations...")
    runs_activations_real = []
    runs_regressors_real = []
    runs_onsets: list[np.ndarray] = []
    runs_offsets: list[np.ndarray] = []

    for run_idx in tqdm(range(n_runs), desc="Extracting"):
        df_run = df_words[df_words["section"] == (run_idx + 1)]
        word_list, onsets, offsets = _extract_valid_word_events(df_run)
        runs_onsets.append(onsets)
        runs_offsets.append(offsets)

        text_file = f"lpp_{lang_lower}_text/text_{lang_lower}_run{run_idx + 1}.txt"
        fulltext = text_zip.read(
            text_file, pwd=b"lessentielestinvisiblepourlesyeux"
        ).decode("utf8")
        fulltext = fulltext.replace("\n", " ")

        layers_words_acts = extract_word_activations_run(
            model=model,
            tokenizer=tokenizer,
            word_list=word_list,
            fulltext_run=fulltext,
            n_layers=n_layers,
            maxlen=maxlen,
            stride=stride,
            lang=lang_lower,
        )

        activations_run = np.array(layers_words_acts[layer])
        runs_activations_real.append(activations_run)

        frame_times = np.arange(fmri_runs[run_idx].shape[0]) * t_r + 0.5 * t_r
        regressor_run = build_regressor_from_activations(
            activations_run, onsets, offsets, frame_times, hrf_model=hrf_model
        )
        n_run = min(regressor_run.shape[0], fmri_runs[run_idx].shape[0])
        regressor_run = regressor_run[:n_run]
        fmri_runs[run_idx] = fmri_runs[run_idx][:n_run]
        regressor_run = (
            regressor_run - regressor_run.mean(axis=0, keepdims=True)
        ) / regressor_run.std(axis=0, keepdims=True)
        runs_regressors_real.append(regressor_run.astype(np.float32))

    text_zip.close()

    # Fit real model once
    print("Fitting real (non-permuted) LORO...")
    loro_real = train_ridge_loro(
        regressors_runs=runs_regressors_real,
        fmri_runs=fmri_runs,
        alphas=list(np.logspace(2, 7, 16)),
    )
    corr_real = loro_real.mean_corr_per_voxel[:n_voxels]

    # ---- Run permutations ----
    print(f"Running {n_permutations} permutations ({permutation_mode})...")
    perm_corrs = []

    for perm_idx in tqdm(range(n_permutations), desc="Permutations"):
        runs_activations_perm = []

        for run_idx in range(n_runs):
            activations_run = runs_activations_real[run_idx].copy()

            if permutation_mode == "shuffle":
                # Shuffle row order (words) within run
                idx_shuffle = np.random.permutation(activations_run.shape[0])
                activations_run = activations_run[idx_shuffle]
            elif permutation_mode == "circular_shift":
                # Circular shift: rotate word order by random amount
                shift_amount = np.random.randint(1, activations_run.shape[0])
                activations_run = np.roll(activations_run, shift_amount, axis=0)
            else:
                raise ValueError(f"Unknown permutation mode: {permutation_mode}")

            runs_activations_perm.append(activations_run)

        # Rebuild design matrices with permuted activations
        runs_regressors_perm = []
        for run_idx in range(n_runs):
            onsets = runs_onsets[run_idx]
            offsets = runs_offsets[run_idx]

            frame_times = np.arange(fmri_runs[run_idx].shape[0]) * t_r + 0.5 * t_r
            regressor_run = build_regressor_from_activations(
                runs_activations_perm[run_idx],
                onsets,
                offsets,
                frame_times,
                hrf_model=hrf_model,
            )
            n_run = min(regressor_run.shape[0], fmri_runs[run_idx].shape[0])
            regressor_run = regressor_run[:n_run]
            regressor_run = (
                regressor_run - regressor_run.mean(axis=0, keepdims=True)
            ) / regressor_run.std(axis=0, keepdims=True)
            runs_regressors_perm.append(regressor_run.astype(np.float32))

        # Fit LORO on permuted data
        loro_perm = train_ridge_loro(
            regressors_runs=runs_regressors_perm,
            fmri_runs=fmri_runs,
            alphas=list(np.logspace(2, 7, 16)),
        )
        perm_corrs.append(loro_perm.mean_corr_per_voxel[:n_voxels])

    perm_corrs = np.array(perm_corrs)  # (n_perm, n_voxels)

    # ---- Compute statistics ----
    perm_mean = perm_corrs.mean(axis=0)  # (n_voxels,)
    perm_95 = np.percentile(perm_corrs, 95, axis=0)  # (n_voxels,)

    # Empirical p-value: fraction of permutations with higher mean voxelwise corr
    empirical_p = (perm_corrs.mean(axis=1) >= corr_real.mean()).mean()

    # ---- Save results ----
    np.save(output_dir / "corr_real.npy", corr_real)
    np.save(output_dir / "corr_permutations.npy", perm_corrs)
    np.save(output_dir / "perm_mean.npy", perm_mean)
    np.save(output_dir / "perm_95.npy", perm_95)

    stats = {
        "model": model_name,
        "layer": int(layer),
        "lang": lang_lower,
        "n_permutations": int(n_permutations),
        "permutation_mode": permutation_mode,
        "seed": int(seed),
        "n_voxels": int(len(corr_real)),
        "real_mean_corr": float(corr_real.mean()),
        "perm_mean_dist_mean": float(perm_corrs.mean(axis=1).mean()),
        "perm_mean_dist_std": float(perm_corrs.mean(axis=1).std()),
        "perm_95_mean": float(perm_95.mean()),
        "empirical_p_value": float(empirical_p),
        "n_perm_better_than_real": int((perm_corrs.mean(axis=1) >= corr_real.mean()).sum()),
    }

    with (output_dir / "permutation_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"Permutation Results: {model_name} (layer {layer}) → {lang_lower}")
    print(f"Real mean corr:         {stats['real_mean_corr']:.4f}")
    print(f"Perm mean dist:         {stats['perm_mean_dist_mean']:.4f} "
          f"(±{stats['perm_mean_dist_std']:.4f})")
    print(f"Empirical p-value:      {stats['empirical_p_value']:.4f}")
    print(f"Permutations ≥ real:    {stats['n_perm_better_than_real']}/{n_permutations}")
    print(f"Results saved to:       {output_dir}")
    print(f"{'='*70}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Permutation baseline for LORO significance testing."
    )
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "fr", "cn"])
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--ds003643_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/permutation")
    parser.add_argument("--n_permutations", type=int, default=20)
    parser.add_argument("--n_voxels", type=int, default=100000)
    parser.add_argument(
        "--permutation_mode",
        type=str,
        default="shuffle",
        choices=["shuffle", "circular_shift"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--access_token", type=str, default=None)

    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = str(Path(__file__).parent.parent / "data")
    if args.ds003643_root is None:
        args.ds003643_root = str(
            Path(__file__).parent.parent / "data" / "openneuro" / "ds003643"
        )

    stats = run_permutation_baseline(
        model_name=args.model,
        lang=args.lang,
        layer=args.layer,
        data_root=args.data_root,
        ds003643_root=args.ds003643_root,
        output_dir=args.output_dir,
        n_permutations=args.n_permutations,
        n_voxels=args.n_voxels,
        permutation_mode=args.permutation_mode,
        seed=args.seed,
        access_token=args.access_token,
    )

    return 0


if __name__ == "__main__":
    exit(main())
