from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
from nilearn.maskers import NiftiMasker


def _standardize(v: np.ndarray) -> np.ndarray:
    """Z-score along axis 0. Ported from llms_brain_lateralization.py."""
    return (v - np.mean(v, axis=0, keepdims=True)) / np.std(v, axis=0, keepdims=True)


def load_fmri_matrix(
    fmri_matrix: str | None,
    fmri_nifti: str | None,
    mask_img: str | None,
    max_voxels: int,
) -> np.ndarray:
    if fmri_matrix:
        y = np.load(fmri_matrix).astype(np.float32)
        return y[:, :max_voxels]

    if not fmri_nifti:
        raise ValueError("Set either dataset.fmri_matrix or dataset.fmri_nifti in config.")

    if not Path(fmri_nifti).exists():
        raise FileNotFoundError(f"fMRI NIfTI not found: {fmri_nifti}")

    masker = NiftiMasker(mask_img=mask_img, standardize=True, detrend=True)
    y = masker.fit_transform(fmri_nifti).astype(np.float32)
    return y[:, :max_voxels]


def load_avg_subject_runs(
    avg_subject_dir: str,
    n_runs: int,
    trim_trs: int = 10,
) -> List[np.ndarray]:
    """Load pre-computed average-subject fMRI runs from joblib .gz files.

    Ported from fit_average_subject.py (Bonnasse-Gahot & Pallier, NeurIPS 2024).

    Loads ``average_subject_run-{0..n_runs-1}.gz``, trims ``trim_trs`` TRs
    from each end, then z-score standardises each run (axis=0).

    Parameters
    ----------
    avg_subject_dir : path to folder containing ``average_subject_run-*.gz``
    n_runs          : number of runs (9 for the LPP corpus)
    trim_trs        : TRs to drop from start and end of each run (default 10,
                      corresponding to 20 s at TR=2 s)
    """
    fmri_runs: List[np.ndarray] = []
    for run in range(n_runs):
        filename = Path(avg_subject_dir) / f"average_subject_run-{run}.gz"
        with open(filename, "rb") as f:
            arr = joblib.load(f)
        if trim_trs > 0:
            arr = arr[trim_trs:-trim_trs]
        arr = _standardize(arr)
        fmri_runs.append(arr.astype(np.float32))
    return fmri_runs


def compute_avg_subject_fmri(
    fmri_data_resampled: str,
    mask_img: str,
    lang: str,
    n_runs: int,
    t_r: float,
    output_dir: str,
) -> None:
    """Average per-subject resampled fMRI across subjects and save .gz files.

    Ported from compute_average_subject_fmri.py (Bonnasse-Gahot & Pallier, NeurIPS 2024).

    For each run: applies NiftiMasker (detrend, standardize, high_pass=1/128 Hz)
    per subject, averages across subjects, z-score standardises the group mean,
    and saves as ``average_subject_run-{run}.gz`` using joblib.

    Parameters
    ----------
    fmri_data_resampled : folder containing per-subject subfolders with .nii.gz runs
    mask_img            : path to the symmetric brain mask (e.g. mask_lpp_en.nii.gz)
    lang                : language code used to identify subject folders (e.g. 'EN')
    n_runs              : number of runs (9 for LPP)
    t_r                 : repetition time in seconds (2.0 for LPP)
    output_dir          : folder where average_subject_run-*.gz will be written
    """
    from tqdm import tqdm

    subject_list = np.sort(
        glob.glob(os.path.join(fmri_data_resampled, f"sub-{lang.upper()}*"))
    )

    os.makedirs(output_dir, exist_ok=True)

    fmri_subs_runs: List[List[np.ndarray]] = []
    for sub_id in tqdm(subject_list, desc="Masking subjects"):
        fmri_imgs_sub = sorted(glob.glob(os.path.join(sub_id, "*.nii.gz")))
        fmri_runs_sub: List[np.ndarray] = []
        for fmri_img in fmri_imgs_sub:
            nifti_masker = NiftiMasker(
                mask_img=mask_img,
                detrend=True,
                standardize=True,
                high_pass=1.0 / 128.0,
                t_r=t_r,
            )
            fmri_runs_sub.append(nifti_masker.fit_transform(fmri_img))
        fmri_subs_runs.append(fmri_runs_sub)

    for run in range(n_runs):
        fmri_mean = np.mean(
            [fmri_subs_runs[s][run] for s in range(len(subject_list))], axis=0
        )
        fmri_mean = _standardize(fmri_mean)
        out_file = os.path.join(output_dir, f"average_subject_run-{run}.gz")
        with open(out_file, "wb") as f:
            joblib.dump(fmri_mean, f, compress=4)
        print(f"Saved run {run}: {out_file} {fmri_mean.shape}")
