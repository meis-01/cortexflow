from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from .io_utils import load_json


DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset": {
        "text_trs": "data/pilot/text_trs.txt",
        "fmri_matrix": "data/pilot/Y.npy",
        "fmri_nifti": None,
        "mask_img": None,
        "max_voxels": 200,
    },
    "features": {
        "model_name": "gpt2",
        "layer": 6,
        "batch_size": 4,
        "max_length": 96,
        "pooling": "mean",
    },
    "design": {
        "delays": [2, 3, 4, 5],
        "standardize": True,
    },
    "training": {
        "alphas": [100.0, 316.0, 1000.0, 3160.0, 10000.0, 31600.0, 100000.0],
        "n_splits": 5,
        "fit_intercept": True,
    },
    "evaluation": {
        "save_predictions": True,
        "hemi_labels": None,
    },
    "output": {
        "dir": "artifacts/paper",
    },
    "seed": 0,
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    cfg = load_json(cfg_path)
    return _deep_update(DEFAULT_CONFIG, cfg)
