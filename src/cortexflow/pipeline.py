from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from .config import load_config
from .design import make_delayed_design
from .features import extract_transformer_features
from .fmri import load_fmri_matrix
from .io_utils import ensure_dir, load_text_lines, save_json
from .metrics import corr_per_voxel, lateralization_index, r2_per_voxel
from .modeling import train_ridge_timeseries_cv


def _align_xy(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(X.shape[0], Y.shape[0])
    return X[:n], Y[:n]


def run_experiment(config_path: str | Path) -> Dict[str, Any]:
    cfg = load_config(config_path)

    text_lines = load_text_lines(Path(cfg["dataset"]["text_trs"]))
    X = extract_transformer_features(
        lines=text_lines,
        model_name=cfg["features"]["model_name"],
        layer=int(cfg["features"]["layer"]),
        batch_size=int(cfg["features"]["batch_size"]),
        max_length=int(cfg["features"]["max_length"]),
        pooling=str(cfg["features"]["pooling"]),
    )

    Y = load_fmri_matrix(
        fmri_matrix=cfg["dataset"].get("fmri_matrix"),
        fmri_nifti=cfg["dataset"].get("fmri_nifti"),
        mask_img=cfg["dataset"].get("mask_img"),
        max_voxels=int(cfg["dataset"]["max_voxels"]),
    )

    Xd, max_delay = make_delayed_design(X, delays=cfg["design"]["delays"])
    Yd = Y[max_delay:]
    Xd, Yd = _align_xy(Xd, Yd)

    train_result = train_ridge_timeseries_cv(
        X=Xd,
        Y=Yd,
        alphas=[float(a) for a in cfg["training"]["alphas"]],
        n_splits=int(cfg["training"]["n_splits"]),
        fit_intercept=bool(cfg["training"]["fit_intercept"]),
        standardize_design=bool(cfg["design"]["standardize"]),
    )

    corr = corr_per_voxel(Yd, train_result.y_pred)
    r2 = r2_per_voxel(Yd, train_result.y_pred)

    out_dir = Path(cfg["output"]["dir"])
    ensure_dir(out_dir)

    np.save(out_dir / "X_features.npy", X)
    np.save(out_dir / "Y_aligned.npy", Yd)
    np.save(out_dir / "Yhat.npy", train_result.y_pred)
    np.save(out_dir / "corr_per_voxel.npy", corr)
    np.save(out_dir / "r2_per_voxel.npy", r2)

    plt.figure(figsize=(6, 4))
    plt.hist(corr, bins=40)
    plt.xlabel("Pearson r")
    plt.ylabel("Voxel count")
    plt.tight_layout()
    plt.savefig(out_dir / "corr_hist.png", dpi=150)
    plt.close()

    metrics: Dict[str, Any] = {
        "best_alpha": train_result.best_alpha,
        "cv_mean_corr": train_result.cv_mean_corr,
        "mean_corr": float(corr.mean()),
        "median_corr": float(np.median(corr)),
        "mean_r2": float(r2.mean()),
        "n_tr": int(Yd.shape[0]),
        "n_vox": int(Yd.shape[1]),
        "n_features": int(Xd.shape[1]),
        "delays": [int(d) for d in cfg["design"]["delays"]],
        "model_name": cfg["features"]["model_name"],
        "layer": int(cfg["features"]["layer"]),
    }

    hemi_labels = cfg["evaluation"].get("hemi_labels")
    if hemi_labels:
        hemi = np.load(hemi_labels).astype(np.int64)[: Yd.shape[1]]
        metrics["lateralization_index"] = float(lateralization_index(corr, hemi))

    save_json(out_dir / "metrics.json", metrics)
    return metrics
