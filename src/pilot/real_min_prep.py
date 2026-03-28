from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import swap_img_hemispheres
from nilearn.masking import compute_multi_epi_mask, intersect_masks
from nilearn.maskers import NiftiMasker


# Paper-exact 4 mm MNI affine (from resample_fmri_data.py,
# Bonnasse-Gahot & Pallier NeurIPS 2024)
_TARGET_AFFINE = np.array(
    [
        [4.0, 0.0, 0.0, -72.0],
        [0.0, 4.0, 0.0, -106.0],
        [0.0, 0.0, 4.0, -64.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
_TARGET_SHAPE = (37, 46, 38)


def _run_key_from_path(path: Path) -> str:
	"""Return a subject-agnostic run key from a BOLD filename."""
	name = path.name
	return re.sub(r"^sub-[^_]+_", "", name)


def _find_bold_by_run(root: Path) -> dict[str, list[Path]]:
	run_map: dict[str, list[Path]] = defaultdict(list)
	for p in sorted(root.glob("sub-*/func/*_bold.nii*")):
		run_map[_run_key_from_path(p)].append(p)
	return run_map


def _trim_trs(arr: np.ndarray, trim_trs: int) -> np.ndarray:
	if trim_trs <= 0:
		return arr
	if arr.shape[0] <= (2 * trim_trs):
		raise ValueError(
			f"Run has only {arr.shape[0]} TRs; cannot trim {trim_trs} TRs from each side."
		)
	return arr[trim_trs:-trim_trs]


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Prepare real fMRI matrix using paper-like group-average preprocessing."
	)
	parser.add_argument("--root", default="data/openneuro/ds003643")
	parser.add_argument("--out_dir", default="data/final")
	parser.add_argument("--max_vox", type=int, default=500)
	parser.add_argument("--tr", type=float, default=2.0, help="TR in seconds")
	parser.add_argument(
		"--high_pass_s",
		type=float,
		default=128.0,
		help="High-pass cutoff in seconds",
	)
	parser.add_argument(
		"--trim_s",
		type=float,
		default=20.0,
		help="Seconds trimmed at start and end of each run after averaging",
	)
	parser.add_argument(
		"--mask_threshold",
		type=float,
		default=0.5,
		help="Threshold passed to compute_multi_epi_mask",
	)
	parser.add_argument(
		"--min_subjects_per_run",
		type=int,
		default=8,
		help="Minimum subjects required for a run key to be included",
	)
	args = parser.parse_args()

	root = Path(args.root)
	out = Path(args.out_dir)
	out.mkdir(parents=True, exist_ok=True)

	run_map = _find_bold_by_run(root)
	if not run_map:
		raise FileNotFoundError(f"No BOLD files found under: {root}")

	run_keys = [
		k for k, files in sorted(run_map.items()) if len(files) >= args.min_subjects_per_run
	]
	if not run_keys:
		raise ValueError(
			"No run keys meet min_subjects_per_run. Lower --min_subjects_per_run or check dataset."
		)

	all_imgs = [str(p) for key in run_keys for p in run_map[key]]
	print(f"Found {len(all_imgs)} BOLD files across {len(run_keys)} run keys.")

	# 1) Group mask at paper-exact 4 mm MNI affine, then symmetrize.
	# Uses intersect_masks + swap_img_hemispheres as in compute_mask.py
	# (Bonnasse-Gahot & Pallier, NeurIPS 2024).
	group_mask = compute_multi_epi_mask(
		all_imgs,
		threshold=args.mask_threshold,
		target_affine=_TARGET_AFFINE,
		target_shape=_TARGET_SHAPE,
	)
	sym_mask = intersect_masks([group_mask, swap_img_hemispheres(group_mask)], threshold=1)
	mask_path = out / "whole_brain_mask_symmetric_4mm.nii.gz"
	nib.save(sym_mask, str(mask_path))

	high_pass_hz = 1.0 / args.high_pass_s
	trim_trs = int(round(args.trim_s / args.tr))

	per_run_avg = []
	run_summary = {}

	# 2) Per run: per-subject NiftiMasker (detrend + standardize + high_pass) ->
	#    average across subjects -> trim -> z-score standardize.
	# Matches compute_average_subject_fmri.py (Bonnasse-Gahot & Pallier, NeurIPS 2024).
	for run_key in run_keys:
		files = run_map[run_key]
		subj_mats = []
		lengths = []
		for p in files:
			masker = NiftiMasker(
				mask_img=sym_mask,
				target_affine=_TARGET_AFFINE,
				target_shape=_TARGET_SHAPE,
				detrend=True,
				standardize=True,
				high_pass=high_pass_hz,
				t_r=args.tr,
			)
			y = masker.fit_transform(str(p)).astype(np.float32)
			subj_mats.append(y)
			lengths.append(y.shape[0])

		min_len = int(min(lengths))
		aligned = np.stack([m[:min_len] for m in subj_mats], axis=0)
		y_avg = aligned.mean(axis=0).astype(np.float32)

		y_avg = _trim_trs(y_avg, trim_trs)
		# Re-standardize group mean (axis=0), matching standardize() in
		# llms_brain_lateralization.py
		y_avg = (
			(y_avg - y_avg.mean(axis=0, keepdims=True))
			/ y_avg.std(axis=0, keepdims=True)
		).astype(np.float32)

		per_run_avg.append(y_avg)
		run_summary[run_key] = {
			"n_subjects": len(files),
			"min_len_tr": min_len,
			"post_trim_tr": int(y_avg.shape[0]),
		}

	y_all = np.concatenate(per_run_avg, axis=0)
	y_all = y_all[:, : args.max_vox].astype(np.float32)

	np.save(out / "Y.npy", y_all)
	(out / "text_trs.txt").write_text("\n".join(["placeholder"] * y_all.shape[0]) + "\n")

	meta = {
		"root": str(root),
		"n_run_keys": len(run_keys),
		"n_files": len(all_imgs),
		"tr": args.tr,
		"high_pass_s": args.high_pass_s,
		"trim_s": args.trim_s,
		"mask_threshold": args.mask_threshold,
		"mask_path": str(mask_path),
		"target_affine": _TARGET_AFFINE.tolist(),
		"target_shape": list(_TARGET_SHAPE),
		"final_shape": [int(y_all.shape[0]), int(y_all.shape[1])],
		"max_vox": args.max_vox,
		"runs": run_summary,
	}
	with (out / "prep_meta.json").open("w", encoding="utf-8") as f:
		json.dump(meta, f, indent=2)

	print(f"Wrote matrix: {out / 'Y.npy'} shape={y_all.shape}")
	print(f"Wrote mask:   {mask_path}")
	print(f"Wrote meta:   {out / 'prep_meta.json'}")


if __name__ == "__main__":
	main()
