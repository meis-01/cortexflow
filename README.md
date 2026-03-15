# cortexflow
CortexFlow is a reproducible MLOps framework for neuroscience experiments that predict brain activity from language model representations. The project demonstrates how modern ML infrastructure (DVC, cloud compute, versioned data, and modular pipelines) can support transparent and scalable computational neuroscience research.

## One-Page Diagram

```mermaid
flowchart TD
	A[Repository\ncode + DVC pipeline + workflow] --> B[make_tr_text.py\nGenerate TR-aligned text\ndata/pilot/text_trs.txt]
	B --> C[extract_gpt2.py\nGPT-2 hidden-state pooling\ndata/pilot/X.npy]
	C --> D[make_synth_bold.py\nSynthetic voxel responses\ndata/pilot/Y.npy]
	C --> E[train_ridge.py\nTimeSeriesSplit + ridge CV\nartifacts/pilot/Yhat.npy]
	D --> E
	D --> F[evaluate.py\nVoxelwise Pearson r\nartifacts/pilot/plots]
	E --> F

	G[real_min_prep.py\nOpenNeuro BOLD -> matrix\ndata/real/Y.npy] -. future real-data path .-> E

	H[dvc.yaml\nDefines stages and artifacts] --> B
	H --> C
	H --> D
	H --> E
	H --> F

	I[GitHub Actions\ndvc-repro.yml] --> J[pip install + dvc pull]
	J --> K[dvc repro]
	K --> B
	K --> C
	K --> D
	K --> E
	K --> F
	K --> L[dvc push on main]

	M[GCP-backed DVC remote] <--> J
	M <--> L

	N[Research objective\nPredict brain activity from LM features] --> C
	N --> E
	N --> F
```

This diagram reflects the repo as it exists today: a synthetic pilot pipeline that is fully reproducible through DVC, plus an initial path toward real fMRI data preparation.
