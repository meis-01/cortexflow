from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from nilearn.glm.first_level import compute_regressor


def make_delayed_design(X: np.ndarray, delays: List[int]) -> Tuple[np.ndarray, int]:
    if not delays:
        return X, 0

    max_delay = int(max(delays))
    blocks = []
    for d in delays:
        d = int(d)
        shifted = np.roll(X, shift=d, axis=0)
        shifted[:d, :] = 0.0
        blocks.append(shifted)

    Xd = np.concatenate(blocks, axis=1)
    return Xd[max_delay:], max_delay


def zscore_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True) + 1e-6
    return (X_train - mu) / sigma, (X_test - mu) / sigma


def build_regressor_from_activations(
    activations: np.ndarray,
    onsets: np.ndarray,
    offsets: np.ndarray,
    frame_times: np.ndarray,
    hrf_model: str = "glover",
) -> np.ndarray:
    """Convolve word-level LLM activations with HRF to produce a run design matrix.

    Ported from fit_average_subject.py (Bonnasse-Gahot & Pallier, NeurIPS 2024).

    Parameters
    ----------
    activations : (n_words, n_neurons)
    onsets      : (n_words,) word onset times in seconds
    offsets     : (n_words,) word offset times in seconds
    frame_times : (n_scans,) TR mid-point times in seconds, i.e.
                  ``np.arange(n_scans) * t_r + 0.5 * t_r``
    hrf_model   : HRF kernel name accepted by nilearn (default "glover")

    Returns
    -------
    np.ndarray of shape (n_scans, n_neurons)
    """
    durations = offsets - onsets
    nn_signals = []
    for amplitudes in activations.T:
        exp_condition = np.array((onsets, durations, amplitudes))
        signal, _ = compute_regressor(exp_condition, hrf_model, frame_times)
        nn_signals.append(signal[:, 0])
    return np.array(nn_signals).T


def build_temporal_design_matrix(
    activations: np.ndarray,
    onsets: np.ndarray,
    offsets: np.ndarray,
    frame_times: np.ndarray,
    temporal_feature_strategy: str,
    hrf_model: str = "glover",
    response_lag_trs: Sequence[int] = (2, 3, 4, 5),
) -> Tuple[np.ndarray, int]:
    """Construct a TR-level design matrix using a named temporal strategy."""
    tr_aligned_activations = build_regressor_from_activations(
        activations,
        onsets,
        offsets,
        frame_times,
        hrf_model=hrf_model,
    )

    if temporal_feature_strategy == "hemodynamic_convolution":
        return tr_aligned_activations, 0

    if temporal_feature_strategy == "lagged_response_window":
        delayed_design, trim_from_start = make_delayed_design(
            tr_aligned_activations,
            list(response_lag_trs),
        )
        return delayed_design, trim_from_start

    raise ValueError(
        "temporal_feature_strategy must be one of: "
        "hemodynamic_convolution, lagged_response_window"
    )
