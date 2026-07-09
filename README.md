# Guiana Ocean Forecasting

Autoregressive **Unet2D** forecasting on the equatorial Guiana margin. All pipelines read ocean fields from a shared Zarr archive, export windowed `.npy` sequences, train with masked L1 or **adv-NO** (L1 + perceptual + adversarial), and evaluate multi-step rollouts.

**Source data:** `s3://oceangen-data/margem_equatorial_tensor.zarr` (override with `GUIANA_ZARR_PATH`).  
**Run notebooks from the project root** unless a subfolder README says otherwise.

---

## Repository map

```
hamid_copy/
├── src/                  # Shared data loaders, export, training, losses
├── scripts/              # Detached export/train workers
├── data/                 # Root fix-window exports
├── Models/  PNG/  logs/  # Root fix-window artifacts
├── Dynamic/              # Lag / timescale analysis (no model training) [To get the limit of dt jumps in sequences it is done for all the 12 windows]
├── Magic_window/         # Fixed catalog-window experiments [select fix windows - These windows are different from 12 windows]
│   └── dynamic_dt/       # Variable forecast-step (dt) sweep [for magicwind_ref train for dt = 3,5,7,...,25,27,30]
├── Shalow/               # Random-crop & 12-window-pool training
│   └── fineTuning/       # Per-window fine-tuning on pool12 [train on fixed weights that was frized from random training on 12 pool windows]
└── Trashes/              # Deprecated notebooks
```

---

## `src/` — shared library

| Module | Role |
|--------|------|
| `data/dataloaders/guiana_margem_tensor.py` | Zarr I/O and tensor access |
| `data/useful_transforms.py` | `RotatedFixedCrop` and spatial transforms |
| `export/fix_window_export.py` | Chunked, resumable fix-window export |
| `export/random_window_export.py` | Random 128×128 crop export |
| `train/adv_no_train.py` | adv-NO training loop, checkpoints, viz |
| `train/losses.py` | L1, Gaussian-center, perceptual, adversarial losses |
| `train/fix_window_train.py` | Baseline Unet2D training |
| `train/random_window_dataset.py` | Window-pool dataset (12-crop rows) |

**Why it exists:** keeps export and training logic out of notebooks so detached scripts and subprojects can reuse the same code.

---

## `scripts/` — background workers

| File | Purpose | Output |
|------|---------|--------|
| `export_fix_window_data.py` | Resumable fix-window export | `{prefix}_*.npy` under `data/` |
| `export_random_window_data.py` | Resumable random-window export | `Shalow/data/{key}/` |
| `train_fix_window.py` | Detached baseline training | `Models/fix_window/` |
| `train_adv_no_fix_window.py` | Detached adv-NO training | `Models/adv_no/{RUN_NAME}/` |
| `train_shalow_adv_no.py` | Detached Shalow training | `Shalow/Models/adv_no/` |
| `dynamic_dt_worker.py` | GPU worker for dynamic_dt train/rollout | `Magic_window/dynamic_dt/` |
| `run_export_detached.sh` | Launch export after disconnect | `logs/export_*.log` |
| `run_train_detached.sh` | Launch baseline train | `logs/train_*.log` |
| `run_train_adv_no_detached.sh` | Launch adv-NO train | `logs/train_adv_no_*.log` |
| `split_uv_data_prefix.py` | Split u/v-only prefixes from multi-channel export | sibling `*_u_*` / `*_v_*` files |

---

## Root — fix-window baseline pipeline

Single **fixed rotated window** (default 128×128 @ top=330, left=70, −45°). The original end-to-end reference path.

| Notebook | Role | Input | Output |
|----------|------|-------|--------|
| `01_export_fix_window_data.ipynb` | Export sequences from Zarr | Zarr | `data/sample_*_fixwin_*` or `sample_*_smallfixwin_*` |
| `02_train_unet2d_fix_window.ipynb` | Baseline Unet2D (masked L1) | `data/sample_*` | `Models/best_model_FixWindow1.pth`, `Models/norm_stats.pt` | (using only l1 the Perceptual and adversial is not implemented here)
| `03_evaluate_fix_window_rollout.ipynb` | 3-step rollout plots | model + test `.npy` | `PNG/output_sample_*.png` |
| `04_train_adv_no_fix_window.ipynb` | adv-NO on fix window | `data/sample_*` | `Models/adv_no/{RUN_NAME}/` | (here the perceptual and adverial is added to loss - all the runs and results with adv_no is using adversial+perceptual loss)
| `05_evaluate_adv_no_rollout.ipynb` | adv-NO rollout + RMSE | model + holdout `.npy` | `PNG/`, metrics in notebook |
| `06_sliding_daily_rollout.ipynb` | Daily sliding anchor from Zarr (no export) | Zarr + checkpoint | `PNG/sliding_*` | for making videos of +15 days and -15 days.
| `06_plot_gaussian_weighted_val_samples.ipynb` | Gaussian-circle validation panels | trained run dir | `{RUN_DIR}/viz/samples_gauss_circles/` | Focus on center
 | `07_train_l1_gauss_circles.ipynb` | Gaussian-center L1 (single field) | `data/sample_*` | `Models/adv_no/{RUN_NAME}/` |
| `08_train_adv_no_single_channel.ipynb` | adv-NO on u or v alone | `*_u_*` / `*_v_*` exports | `Models/adv_no/{RUN_NAME}/` |

**Artifacts:** `data/` · `Models/` · `PNG/` · `logs/`  
**Model:** `gUnet.py` (root Unet2D definition).

---

## `Dynamic/` — window dynamics analysis

Offline **lag and memory analysis** on existing exports. No model training.

| File | Role | Input | Output |
|------|------|-------|--------|
| `Dynamic_data_selected_windows.ipynb` | Per-window lag curves (ρ, MI, integral timescale) | `data/sample_*` or catalog exports | `Dynamic/data/fixwin_{key}/` or `Dynamic/data/window_{NN}_{key}/` |
| `Dynamic_pointwise_selected_windows.ipynb` | Per-pixel autocorrelation maps | pool12 / catalog exports | `Dynamic/pointwise_data/window_{NN}_{key}/` |
| `window_dynamics.py` | Lag metrics implementation | — | — |
| `pointwise_dynamics.py` | Grid-cell autocorrelation / spectrum | — | — |

**Why it exists:** quantify temporal decorrelation per window before choosing forecast horizons and training windows.

---

## `Magic_window/` — fixed catalog windows

Twelve named **fixed windows** (reference, coastal slide, bathymetry, rotation variants). Full export → train → rollout pipeline isolated under this folder.

### Support modules

| File | Role |
|------|------|
| `window_selection.ipynb` / `Magic_window_window_selection.ipynb` | Preview and pick window geometry |find the coordinates of 12 windows.
| `window_catalog.py` | Canonical 12-window definitions |
| `window_paths.py` | Path helpers per `EXPORT_KEY` |
| `run_registry.py` | Run naming and comparison index |
| `compare_rollout_results.py` | Aggregate rollout JSON across runs | Compare the results of roolout- Heatmap of RMSE and Skills
| `Magic_window_visualize.ipynb` | Lat/lon/bathymetry overview of reference patch |

### Pipeline notebooks

| Notebook | Role | Input | Output |
|----------|------|-------|--------|
| `01_export_magic_window.ipynb` | Export train/val/test per window | Zarr | `Magic_window/data/{export_key}/sample_*` |
| `04_train_adv_no_magic_window.ipynb` | adv-NO training | `data/{export_key}/` | `Magic_window/Models/adv_no/{RUN_NAME}/` |
| `04_train_adv_no_magic_window_gaussian.ipynb` | adv-NO + Gaussian-center loss | same | same (gaussian tag) |
| `05_evaluate_adv_no_magic_window_rollout.ipynb` | Cross-eval (Shalow → magic window) | Shalow checkpoint + magic test | `Magic_window/Test&Rollout/{run}__on__{key}/` | Trained on randowly 12poll windows and evaluate on magic window only.
| `05_evaluate_adv_no_magic_window_rollout_gaussian.ipynb` | Rollout with Gaussian-circle viz | Shalow checkpoint | `Magic_window/Test&Rollout/` |
| `06_sliding_daily_rollout_magic_window.ipynb` | Sliding daily rollout from Zarr | Zarr + magic checkpoint | `Magic_window/PNG/` |
| `07_compare_rollout_results.ipynb` | Cross-window comparison tables/plots | all `rollout_results.json` | `Magic_window/Test&Rollout/_comparison/` |

**Artifacts:** `Magic_window/data/` · `Models/adv_no/` · `Test&Rollout/` · `PNG/` · `logs/`

---

## `Magic_window/dynamic_dt/` — variable forecast step

Trains and evaluates the **same window at multiple frame spacings** (dt = 3–30 days). Uses 5 conditioning frames → 1 prediction (u, v).

| File / notebook | Role | Input | Output |
|-----------------|------|-------|--------|
| `01_export_dynamic_dt.ipynb` | Export all dt values (resumable) | Zarr | `dynamic_dt/data/{export_key}/dt{hours}/sample_*` |
| `04_train_adv_no_dynamic_dt.ipynb` | Parallel adv-NO per dt | exports from 01 | `dynamic_dt/Models/adv_no/{RUN_NAME}/` |
| `05_evaluate_rollout_dynamic_dt.ipynb` | 3-step rollout per dt | model + 2023 test | `dynamic_dt/Test&Rollout/{RUN_NAME}/rollout_results.json` |
| `06_plot_rollout_dynamic_dt.ipynb` | Speed-RMSE vs dt curves | rollout JSONs | `dynamic_dt/Test&Rollout/_comparison/` |
| `train_runner.py` | Build configs, launch training | — | — |
| `rollout_core.py` | Rollout evaluation logic | — | — |

**Windows:** `magicwin_reference` (axis-aligned) · `smallfixwin` (rotated −45°).  
**Logs:** `dynamic_dt/logs/`

---

## `Shalow/` — random crops & window pools

Trains on **spatially varying** windows: random 128×128 crops or all 12 catalog windows cached per time anchor. Self-contained — all paths relative to `Shalow/`.

### Export notebooks

| Notebook | Strategy | Output dir |
|----------|----------|------------|
| `01_export_random_window.ipynb` | Random crop pool (noon, K crops/anchor) | `Shalow/data/random_win_128/` |
| `01_export_random_window_uvsshbathy.ipynb` | Random crop, multi-channel (u,v,ssh,bathy) | `Shalow/data/{EXPORT_KEY}/` |
| `01_export_select_window_pool12.ipynb` | 12 catalog windows per snapshot | `Shalow/data/select_window_12/` |
| `01_export_magic_window_pool12.ipynb` | Legacy pool12 variant | `Shalow/data/` |
| `01_export_magic_window_fix.ipynb` | Single magic-window export (eval-focused) | `Magic_window/data/{key}/` or `Shalow/Magic_window/data/` |

### Train & evaluate

| Notebook | Role | Output |
|----------|------|--------|
| `04_Sh_train_magic_window.ipynb` | adv-NO on random-window pool | `Shalow/Models/adv_no/{RUN_NAME}/` |
| `04_Sh_train_k1_random_12_l1+advno_1.ipynb` | adv-NO on 12-crop pool (k=1) | same |
| `04_Sh_train_select_window_12.ipynb` | L1 on catalog pool12 (k slots/epoch) | same |
| `05_Sh_evaluate_magic_window_rollout.ipynb` | Rollout on 2023 random-window test | `Shalow/Test&Rollout/{RUN_NAME}__on__{key}/` |
| `05_Sh_evaluate_magic_window_rollout_pool12_l1random_k1.ipynb` | Cross-eval pool12 model on magic window | `Shalow/Test&Rollout/` |
| `05_Sh_test_select_window_12.ipynb` | Per-window metrics on 2023 pool12 test | `Shalow/Test&Rollout/.../by_window/` |

**Local modules:** `gUnet.py`, `Shalow_gUnet.py`, `Shalow_gUnet_xs.py`, `run_registry.py`, `window_paths.py`  
**Artifacts:** `Shalow/data/` · `Models/adv_no/` · `Test&Rollout/` · `PNG/` · `logs/`

---

## `Shalow/fineTuning/` — per-window specialization

Fine-tune a **pool12-pretrained** model on a single catalog window slice.

| Notebook | Role | Input | Output |
|----------|------|-------|--------|
| `04_Sh_finetune_catalog_window.ipynb` | Fine-tune on one window (W1–W12) | pool12 export + pretrained weights | `Shalow/Models/fineTuning/{RUN_NAME}/` |
| `05_Sh_finetune_eval_rollout.ipynb` | Rollout on that window's 2023 test slice | fine-tuned checkpoint | `fineTuning/Test&Rollout/{RUN_NAME}__on__window_XX/` |
| `comparision.ipynb` | Fine-tune vs pooled baseline (+15d) | both rollout dirs | comparison plots in notebook |

**Why it exists:** test whether window-specific adaptation beats a single model trained on all 12 windows.

---

## `Trashes/`

Deprecated exploration notebooks (`autoReg_guiana`, `explore_guiana_data`, etc.). Not part of any active pipeline.  
`data_fix_window_4years.ipynb` at root is an older copy of `01_export_fix_window_data.ipynb`.

---

## Typical workflows

```
Fix-window baseline
  01 → data/ → 04 → Models/adv_no/ → 05 → PNG/

Magic window (one catalog window)
  window_selection → 01 → Magic_window/data/ → 04 → Models/ → 05 → Test&Rollout/

Shalow random window
  01 → Shalow/data/ → 04_Sh_train → Shalow/Models/ → 05_Sh_eval → Test&Rollout/

Dynamic dt sweep
  dynamic_dt/01 → data/dt*/ → 04 → Models/ → 05 → Test&Rollout/ → 06 → _comparison/

Cross-benchmark
  Shalow train → Magic_window/05 (cross-eval) → 07_compare_rollout_results
```

---

## Prerequisites

- Python: `numpy`, `torch`, `matplotlib`, `joblib`, `zarr`, `pandas`, `scipy`, `einops`, Jupyter
- AWS credentials for S3 Zarr (or local `GUIANA_ZARR_PATH`)
- GPU recommended for training notebooks (device IDs set per notebook)
