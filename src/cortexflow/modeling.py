from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from .design import zscore_train_test
from .metrics import corr_per_voxel


@dataclass
class TrainingResult:
    best_alpha: float
    cv_mean_corr: float
    y_pred: np.ndarray


@dataclass
class LoroResult:
    """Per-voxel mean correlation from leave-one-run-out CV."""
    mean_corr_per_voxel: np.ndarray          # (n_voxels,)
    best_alphas: List[float] = field(default_factory=list)  # one per fold


def train_ridge_timeseries_cv(
    X: np.ndarray,
    Y: np.ndarray,
    alphas: List[float],
    n_splits: int,
    fit_intercept: bool,
    standardize_design: bool,
) -> TrainingResult:
    cv = TimeSeriesSplit(n_splits=n_splits)
    alpha_scores = []

    for alpha in alphas:
        fold_scores = []
        for tr_idx, te_idx in cv.split(X):
            Xtr, Xte = X[tr_idx], X[te_idx]
            Ytr, Yte = Y[tr_idx], Y[te_idx]

            if standardize_design:
                Xtr, Xte = zscore_train_test(Xtr, Xte)

            model = Ridge(alpha=float(alpha), fit_intercept=fit_intercept)
            model.fit(Xtr, Ytr)
            pred = model.predict(Xte)
            fold_scores.append(float(corr_per_voxel(Yte, pred).mean()))

        alpha_scores.append(float(np.mean(fold_scores)))

    best_idx = int(np.argmax(alpha_scores))
    best_alpha = float(alphas[best_idx])

    Xfit = X
    if standardize_design:
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True) + 1e-6
        Xfit = (X - mu) / sigma

    final_model = Ridge(alpha=best_alpha, fit_intercept=fit_intercept)
    final_model.fit(Xfit, Y)
    y_pred = final_model.predict(Xfit).astype(np.float32)

    return TrainingResult(
        best_alpha=best_alpha,
        cv_mean_corr=float(max(alpha_scores)),
        y_pred=y_pred,
    )


def train_ridge_loro(
    regressors_runs: List[np.ndarray],
    fmri_runs: List[np.ndarray],
    alphas: List[float],
) -> LoroResult:
    """Leave-one-run-out Ridge regression with nested CV for alpha selection.

    Ported from fit_average_subject.py (Bonnasse-Gahot & Pallier, NeurIPS 2024).

    Parameters
    ----------
    regressors_runs : list of n_runs arrays, each (n_scans_run, n_neurons)
    fmri_runs       : list of n_runs arrays, each (n_scans_run, n_voxels)
    alphas          : candidate regularisation values, e.g. np.logspace(2, 7, 16)

    Returns
    -------
    LoroResult with mean_corr_per_voxel averaged across held-out runs.
    """
    n_runs = len(regressors_runs)
    n_voxels = fmri_runs[0].shape[1]

    corr_runs: List[List[float]] = []
    best_alphas: List[float] = []

    for run_test in range(n_runs):
        runs_train = np.setdiff1d(np.arange(n_runs), run_test)
        x_train = np.vstack([regressors_runs[r] for r in runs_train])
        x_test = regressors_runs[run_test]
        y_train = np.vstack([fmri_runs[r] for r in runs_train])
        y_test = fmri_runs[run_test]

        # --- nested CV: hold out one more run to select alpha ---
        run_val = runs_train[0]
        runs_train_val = np.setdiff1d(runs_train, run_val)
        x_train_val = np.vstack([regressors_runs[r] for r in runs_train_val])
        x_val = regressors_runs[run_val]
        y_train_val = np.vstack([fmri_runs[r] for r in runs_train_val])
        y_val = fmri_runs[run_val]

        corr_val: List[List[float]] = []
        for alpha in alphas:
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(x_train_val, y_train_val)
            y_pred = model.predict(x_val)
            corr_tmp = [float(np.corrcoef(y_val[:, i], y_pred[:, i])[0, 1]) for i in range(n_voxels)]
            corr_val.append(corr_tmp)

        idx_best = int(np.argmax(np.mean(corr_val, axis=1)))
        best_alpha = float(alphas[idx_best])
        best_alphas.append(best_alpha)
        # --- end nested CV ---

        model = Ridge(alpha=best_alpha, fit_intercept=False)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        corr_tmp = [float(np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]) for i in range(n_voxels)]
        corr_runs.append(corr_tmp)

    return LoroResult(
        mean_corr_per_voxel=np.mean(corr_runs, axis=0).astype(np.float32),
        best_alphas=best_alphas,
    )
