#!/usr/bin/env python
"""
LORO (Leave-One-Run-Out) Ridge regression experiment runner.

Ported from fit_average_subject.py (Bonnasse-Gahot & Pallier, NeurIPS 2024).
Generalized for cortexflow reproducibility and extensibility.

Usage:
    python run_loro_experiment.py \\
        --model gpt2 \\
        --lang en \\
        --layer 6 \\
        --data_root ~/Projects/llms_brain_lateralization \\
        --ds003643_root ~/Projects/cortexflow/data/openneuro/ds003643 \\
        --output_dir results/loro_gpt2_en_layer6 \\
        --n_voxels 100000
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import zipfile
from tqdm import tqdm

# Cortexflow pipeline imports
from cortexflow.design import build_regressor_from_activations
from cortexflow.features import (
    extract_word_activations_run,
    load_llm_model,
)
from cortexflow.fmri import load_avg_subject_runs
from cortexflow.modeling import train_ridge_loro


def load_word_onsets_offsets(
    ds003643_root: str,
    lang: str,
) -> Tuple[pd.DataFrame, str]:
    """Load word-level timing info from OpenNeuro annotation CSV.

    Returns (dataframe, lang_upper) where dataframe has columns:
        section, onset, offset, word
    """
    lang_upper = lang.upper()
    csv_path = Path(ds003643_root) / "annotation" / lang_upper / f"lpp{lang_upper}_word_information.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df = df.reset_index(drop=True)  # Reset index to avoid drop issues

    required_cols = {"word", "onset", "offset", "section"}
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise ValueError(
            f"Annotation CSV missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    
    # Apply paper's ad-hoc corrections (safely, only if rows exist)
    if lang == "en":
        rows_to_drop = [i for i in [3919, 6775, 6781] if i < len(df)]
        if rows_to_drop:
            df = df.drop(rows_to_drop).reset_index(drop=True)
    elif lang == "fr":
        # Only apply corrections if rows exist
        for idx, val in [(3332, "de"), (3379, "trois"), (3405, "trois"), (4587, "l"), 
                         (5325, "la"), (5326, "première"), (5328, "habitée"), (11257, "À"), (12249, "il")]:
            if idx < len(df):
                df.loc[idx, "word"] = val
        rows_to_drop = [i for i in [338, 1204, 3333] if i < len(df)]
        if rows_to_drop:
            df = df.drop(rows_to_drop).reset_index(drop=True)
    elif lang == "cn":
        pass
    else:
        raise ValueError(f"Unknown language: {lang}")
    
    return df, lang_upper


def load_text_zip(
    data_root: str,
    lang: str,
) -> zipfile.ZipFile:
    """Open the encrypted text ZIP file."""
    text_zip = Path(data_root) / f"lpp_{lang}_text.zip"
    if not text_zip.exists():
        raise FileNotFoundError(f"Text ZIP not found: {text_zip}")
    return zipfile.ZipFile(text_zip, "r")


def _extract_valid_word_events(df_run: pd.DataFrame) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Keep only valid word rows to match the original paper preprocessing."""
    word_list: List[str] = []
    onsets: List[float] = []
    offsets: List[float] = []
    for word, onset, offset in zip(df_run["word"], df_run["onset"], df_run["offset"]):
        if isinstance(word, str) and word != " ":
            word_list.append(word)
            onsets.append(float(onset))
            offsets.append(float(offset))
    return word_list, np.array(onsets), np.array(offsets)


def run_experiment(
    model_name: str,
    lang: str,
    layer: int,
    output_dir: str = "results/loro",
    data_root: Optional[str] = None,
    ds003643_root: Optional[str] = None,
    n_runs: int = 9,
    t_r: float = 2.0,
    hrf_model: str = "glover",
    n_voxels: int = 100000,
    alphas: Optional[List[float]] = None,
    access_token: Optional[str] = None,
) -> Dict[str, float]:
    """Run full LORO Ridge regression experiment.

    Parameters
    ----------
    model_name : str
        HuggingFace model (e.g., "gpt2", "openai-community/gpt2-large")
    lang : str
        Language code ("en", "fr", "cn")
    layer : int
        Layer index (0 = embedding, 1..n_layers = transformer layers)
    data_root : str
        Path to llms_brain_lateralization repo (containing lpp_*.zip, lpp_*_average_subject/)
    ds003643_root : str
        Path to OpenNeuro ds003643 dataset (containing annotation/, derivatives/)
    output_dir : str
        Output directory for results
    n_runs : int
        Number of runs (9 for LPP)
    t_r : float
        Repetition time in seconds (2.0 for LPP)
    hrf_model : str
        HRF kernel ("glover", etc.)
    n_voxels : int
        Max voxels to keep
    alphas : list of float
        Ridge regularization values (default: logspace(2, 7, 16))
    access_token : str
        HuggingFace access token for gated models

    Returns
    -------
    dict with keys:
        mean_corr, median_corr, n_voxels_actual, n_features, best_alphas_per_fold, ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default paths: use cortexflow-relative locations
    if data_root is None:
        data_root = str(Path(__file__).parent.parent / "data")
    if ds003643_root is None:
        ds003643_root = str(Path(__file__).parent.parent / "data" / "openneuro" / "ds003643")

    if alphas is None:
        alphas = list(np.logspace(2, 7, 16))

    lang_lower = lang.lower()
    assert lang_lower in ["en", "fr", "cn"], f"Unsupported language: {lang}"

    # ---- Load fMRI ----
    print(f"\n[1/4] Loading avg-subject fMRI runs for {lang_lower}...")
    avg_subject_dir = Path(data_root) / f"lpp_{lang_lower}_average_subject"
    if not avg_subject_dir.exists():
        raise FileNotFoundError(f"Average subject dir not found: {avg_subject_dir}")

    fmri_runs = load_avg_subject_runs(
        avg_subject_dir=str(avg_subject_dir),
        n_runs=n_runs,
        trim_trs=10,  # 20 seconds at TR=2s
    )
    print(f"   Loaded {len(fmri_runs)} runs, shape per run: {fmri_runs[0].shape}")

    # ---- Load LLM ----
    print(f"\n[2/4] Loading LLM: {model_name}, layer {layer}...")
    model, tokenizer, n_layers, maxlen, stride = load_llm_model(
        model_name, access_token=access_token
    )
    print(f"   Model has {n_layers} layers (incl. embedding)")

    # ---- Load word annotations & text ----
    print(f"\n[3/4] Extracting word-level LLM activations for {lang_lower}...")
    df_words, lang_upper = load_word_onsets_offsets(ds003643_root, lang_lower)
    text_zip = load_text_zip(data_root, lang_lower)

    # Extract activations per run
    runs_activations = []
    runs_regressors = []

    for run_idx in tqdm(range(n_runs), desc="Runs"):
        # Get words for this run from annotation CSV
        df_run = df_words[df_words["section"] == (run_idx + 1)]
        word_list, onsets, offsets = _extract_valid_word_events(df_run)

        # Get full text for this run
        text_file = f"lpp_{lang_lower}_text/text_{lang_lower}_run{run_idx + 1}.txt"
        fulltext = text_zip.read(
            text_file, pwd=b"lessentielestinvisiblepourlesyeux"
        ).decode("utf8")
        fulltext = fulltext.replace("\n", " ")

        # Extract word-level activations
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

        # activations_run: (n_words, n_neurons)
        activations_run = np.array(layers_words_acts[layer])  # select layer
        runs_activations.append(activations_run)

        # Convolve with HRF
        frame_times = np.arange(fmri_runs[run_idx].shape[0]) * t_r + 0.5 * t_r
        regressor_run = build_regressor_from_activations(
            activations_run, onsets, offsets, frame_times, hrf_model=hrf_model
        )

        # fMRI runs are already trimmed by load_avg_subject_runs(trim_trs=10).
        # Keep regressor at the same run length and align defensively.
        n_run = min(regressor_run.shape[0], fmri_runs[run_idx].shape[0])
        regressor_run = regressor_run[:n_run]
        fmri_runs[run_idx] = fmri_runs[run_idx][:n_run]
        regressor_run = (
            regressor_run - regressor_run.mean(axis=0, keepdims=True)
        ) / regressor_run.std(axis=0, keepdims=True)
        runs_regressors.append(regressor_run.astype(np.float32))

    text_zip.close()

    print(f"   Extracted {len(runs_regressors)} runs of design matrices")
    print(f"   Design matrix shape per run: {runs_regressors[0].shape}")

    # ---- Fit LORO ----
    print(f"\n[4/4] Fitting LORO Ridge with nested CV...")
    loro_result = train_ridge_loro(
        regressors_runs=runs_regressors,
        fmri_runs=fmri_runs,
        alphas=alphas,
    )

    corr = loro_result.mean_corr_per_voxel[:n_voxels]

    # ---- Save results ----
    print(f"\n[Results] Saving to {output_dir}")
    np.save(output_dir / "corr_per_voxel.npy", corr)

    metrics = {
        "model": model_name,
        "layer": int(layer),
        "lang": lang_lower,
        "n_voxels": int(len(corr)),
        "n_features": int(runs_regressors[0].shape[1]),
        "n_runs": int(n_runs),
        "mean_corr": float(np.mean(corr)),
        "median_corr": float(np.median(corr)),
        "std_corr": float(np.std(corr)),
        "min_corr": float(np.min(corr)),
        "max_corr": float(np.max(corr)),
        "best_alphas_per_fold": [float(a) for a in loro_result.best_alphas],
        "alphas_tested": [float(a) for a in alphas],
        "hrf_model": hrf_model,
        "t_r": float(t_r),
    }

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"Experiment: {model_name} (layer {layer}) → {lang_lower}")
    print(f"Mean correlation: {metrics['mean_corr']:.4f}")
    print(f"Median correlation: {metrics['median_corr']:.4f}")
    print(f"Voxels: {metrics['n_voxels']}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run LORO Ridge experiment on average-subject fMRI."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name (e.g., gpt2, gpt2-large, opt-350m)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=["en", "fr", "cn"],
        help="Language (en, fr, cn)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=6,
        help="Layer index (0=embedding, 1+= transformer layers)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to data/ folder (default: cortexflow/data)",
    )
    parser.add_argument(
        "--ds003643_root",
        type=str,
        default=None,
        help="Path to OpenNeuro ds003643 (default: cortexflow/data/openneuro/ds003643)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/loro",
        help="Output directory for results",
    )
    parser.add_argument(
        "--n_voxels",
        type=int,
        default=100000,
        help="Maximum voxels to keep (for memory)",
    )
    parser.add_argument(
        "--hrf_model",
        type=str,
        default="glover",
        help="HRF kernel (glover, spm, canonical)",
    )
    parser.add_argument(
        "--access_token",
        type=str,
        default=None,
        help="HuggingFace access token for gated models",
    )

    args = parser.parse_args()

    metrics = run_experiment(
        model_name=args.model,
        lang=args.lang,
        layer=args.layer,
        data_root=args.data_root,
        ds003643_root=args.ds003643_root,
        output_dir=args.output_dir,
        n_voxels=args.n_voxels,
        hrf_model=args.hrf_model,
        access_token=args.access_token,
    )

    return 0


if __name__ == "__main__":
    exit(main())
