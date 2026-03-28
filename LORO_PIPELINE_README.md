# CortexFlow LORO Pipeline – Local Reproduction

Generalized, reproducible runners for **Leave-One-Run-Out (LORO) Ridge regression** targeting the NeurIPS 2024 paper: *"fMRI predictors based on language models of increasing complexity recover brain left lateralization"* (Bonnasse-Gahot & Pallier).

All code integrates exact paper-ported implementations from `llms_brain_lateralization` repo while maintaining clean, extensible architecture in cortexflow.

---

## Quick Start (EEML Abstract Deadline: March 31, 2026)

### 1. Run single model (English, GPT-2, layer 6)

```powershell
cd C:\Users\meisa\Projects\cortexflow

# Run experiment (~30–60 min)
# Data paths default to cortexflow/data/ automatically
.\.env\Scripts\python src/run_loro_experiment.py `
    --model gpt2 `
    --lang en `
    --layer 6 `
    --output_dir results/loro/gpt2_en_layer6
```

Results will be in `results/loro/gpt2_en_layer6/`:
- `corr_per_voxel.npy` — correlations (1D array)
- `metrics.json` — summary stats (mean_corr, median_corr, etc.)

### 2. Run permutation baseline (~2–3 hours for 20 permutations)

```powershell
.\.env\Scripts\python src/run_loro_permutation.py `
    --model gpt2 `
    --lang en `
    --layer 6 `
    --n_permutations 20 `
    --output_dir results/permutation/gpt2_en_layer6
```

Results:
- `corr_real.npy`, `corr_permutations.npy` — arrays
- `permutation_stats.json` — includes `empirical_p_value`

### 3. Run batch (all languages)

```powershell
.\.env\Scripts\python src/run_loro_batch.py `
    --config batch_config_eeml.json `
    --output_dir results/loro_batch_eeml
```

Runs GPT-2 layer 6 for EN, FR, CN sequentially. Outputs to `results/loro_batch_eeml/` with summary in `batch_results.json`.

---

## Architecture

### Generalized Pipeline Functions (in `cortexflow/`)

All exact paper implementations refactored for reuse:

- **`design.py:build_regressor_from_activations()`** — HRF convolution
- **`modeling.py:train_ridge_loro()`** — Leave-one-run-out CV with nested alpha selection
- **`fmri.py:load_avg_subject_runs()`** — Load pre-computed average-subject data
- **`features.py:load_llm_model()`, `extract_word_activations_run()`** — Word-level LLM extraction (GPT-2, OPT, Mistral, Llama, etc.)

### Runners (in `src/`)

1. **`run_loro_experiment.py`** — Single experiment
   - Loads pre-computed avg-subject fMRI
   - Extracts word activations from any HuggingFace model
   - Fits LORO with nested CV for alpha selection
   - Outputs: `corr_per_voxel.npy`, `metrics.json`

2. **`run_loro_batch.py`** — Batch runner
   - CLI: `--experiments MODEL:LANG:LAYER [...]`
   - Or config: `--config batch_config.json`
   - Sequential execution, summary aggregation

3. **`run_loro_permutation.py`** — Permutation baseline
   - Shuffles word order within run (or circular-shift)
   - Refits LORO on permuted design
   - Computes empirical p-value
   - Outputs: `permutation_stats.json`, `corr_permutations.npy`

---

## Data Requirements

### ✅ Pre-computed (all in `cortexflow/data/`)

No external dependencies! All data is self-contained:
- `data/lpp_{en,fr,cn}_average_subject/` — 9 runs, pre-averaged across subjects
- `data/lpp_{en,fr,cn}_text.zip` — full text per run (encrypted)
- `data/openneuro/ds003643/` — full dataset + annotations

---

## Customization

### Use different model

```powershell
.\.env\Scripts\python src/run_loro_experiment.py `
    --model gpt2-large `
    --lang en `
    --layer 10
```

Supported: `gpt2`, `gpt2-medium`, `gpt2-large`, `opt-350m`, `opt-1.3b`, `opt-2.7b`, `Mistral-7B`, `Llama-2-7b`, etc.

### Change output directory structure

```powershell
# All results in one output dir
.\.env\Scripts\python src/run_loro_experiment.py `
    --output_dir my_results
```

### Reduce memory usage (larger models)

```powershell
.\.env\Scripts\python src/run_loro_experiment.py `
    --n_voxels 10000   # default 100000
    --model opt-2.7b
```

### Permutation modes

```powershell
# Shuffle: random word reordering
.\.env\Scripts\python src/run_loro_permutation.py `
    --permutation_mode shuffle

# Circular shift: rotate word sequence
.\.env\Scripts\python src/run_loro_permutation.py `
    --permutation_mode circular_shift
```

---

## Results Interpretation

### Real experiment output (`metrics.json`)

```json
{
  "model": "gpt2",
  "layer": 6,
  "lang": "en",
  "mean_corr": 0.245,
  "median_corr": 0.189,
  "std_corr": 0.341,
  "n_voxels": 100000,
  "best_alphas_per_fold": [3.16, 3.16, 3.16, ..., 3.16]
}
```

Paper comparison (EN, GPT-2, layer 6, ~2000 voxels): **mean_corr ≈ 0.24**

### Permutation output (`permutation_stats.json`)

```json
{
  "real_mean_corr": 0.245,
  "perm_mean_dist_mean": 0.018,
  "empirical_p_value": 0.0,
  "n_perm_better_than_real": 0
}
```

- **empirical_p_value = 0.0** → None of 20 permutations matched real performance → highly significant

---

## For EEML Abstract

Typical workflow (total time ~4 hours):

```powershell
# 1. Run 3 languages (GPT-2, layer 6) — ~1.5 hours
# All paths default to cortexflow/data automatically
.\.env\Scripts\python src/run_loro_batch.py `
    --config batch_config_eeml.json `
    --output_dir results/eeml_final

# 2. Run permutation baseline (1 language, 20 perms) — ~2.5 hours
.\.env\Scripts\python src/run_loro_permutation.py `
    --model gpt2 `
    --lang en `
    --layer 6 `
    --n_permutations 20 `
    --output_dir results/eeml_final/english_permutation

# 3. Generate table in Jupyter:
#    - Load corr.npy, metrics.json from batch results
#    - Load permutation_stats.json
#    - Create 2×4 table: Model | Mean r | Median r | Null mean | p-value
```

Then write 2-page LaTeX abstract using results as proof of successful reproduction.

---

## Extension Ideas

1. **Different languages:** swap `--lang fr` or `--lang cn`
2. **More models:** `--model opt-1.3b`, `--model Mistral-7B`
3. **Batch across languages & models:** edit `batch_config_eeml.json`
4. **Individual subject fitting:** implement `run_individual_subject.py` (uses cortexflow pipeline on per-subject data)
5. **ROI analysis:** load `mask_lpp_*.nii.gz` from original repo, subset voxels by lateralization
6. **Layer-by-layer comparison:** loop over layers 0–12, plot correlation vs. layer depth

---

## Troubleshooting

**"Annotation file not found"**
- Ensure `data/openneuro/ds003643/annotation/EN/` exists
- Check path: `ds003643_root` argument

**"Annotation CSV loads as one line like /annex/objects/..."**
- This means the file is a git-annex pointer stub, not real CSV content.
- Re-download real annotation files directly from OpenNeuro:
    - `Invoke-WebRequest -Uri "https://s3.amazonaws.com/openneuro.org/ds003643/annotation/EN/lppEN_word_information.csv" -OutFile "data/openneuro/ds003643/annotation/EN/lppEN_word_information.csv"`
    - `Invoke-WebRequest -Uri "https://s3.amazonaws.com/openneuro.org/ds003643/annotation/FR/lppFR_word_information.csv" -OutFile "data/openneuro/ds003643/annotation/FR/lppFR_word_information.csv"`
    - `Invoke-WebRequest -Uri "https://s3.amazonaws.com/openneuro.org/ds003643/annotation/CN/lppCN_word_information.csv" -OutFile "data/openneuro/ds003643/annotation/CN/lppCN_word_information.csv"`

**"lpp_en_average_subject not found"**
- Ensure `data/lpp_en_average_subject/` exists in this repo.
- If needed, set `--data_root` explicitly to your local `data/` directory.

**Memory error during extraction**
- Reduce `--n_voxels` (e.g., 10000 instead of 100000)
- Or reduce batch size in `features.py` if needed

**ValueError during word-matching**
- Rare; usually indicates mismatch in text/annotation alignment
- May require ad-hoc fixes (see `extract_word_activations_run()`)

---

## References

- **Paper:** Bonnasse-Gahot & Pallier, NeurIPS 2024, https://openreview.net/...  
- **Original code:** https://github.com/...llms_brain_lateralization  
- **OpenNeuro dataset:** https://openneuro.org/datasets/ds003643  

---

**Clean, reproducible, extensible.** Ready for EEML 2026! 🎯
