# Supplemental Materials

This directory contains the complete source code, configurations, training logs, and validation results for reproducing the experiments described in the paper.

## Directory Structure

```
supplemental_materials/
├── configs/             # Training configuration files (YAML)
├── models/              # Neural network architecture implementations
├── training/            # Training scripts (dataset, trainers)
├── validation/          # Validation scripts and shell launchers
├── notebooks/           # Jupyter notebooks (training, plotting, visualization)
├── logs/                # Training logs (per-epoch loss values)
├── images/              # Output figures (loss plots, projection montages)
├── results/
│   ├── flow_matching/   # Per-patient validation metrics (flow matching models)
│   └── regression/      # Per-patient validation metrics (regression baselines)
└── README.md
```

## Models

We compare **Flow Matching** models against **Regression** baselines and a **Diffusion** model for contrast-to-native CT image translation.

### Flow Matching Models

| Model                     | Config                                  | Description                          |
|---------------------------|-----------------------------------------|--------------------------------------|
| **TimeResNet** (ours)     | `flow_match_mynet.yaml`                 | Full model: time conditioning + attention |
| TimeResNet w/o attention  | `flow_match_mynet_wo_attention.yaml`    | Ablation: no self-attention layers    |
| TimeResNet w/o time       | `flow_match_mynet_wo_time.yaml`         | Ablation: no time embedding           |
| TimeResNet w/o both       | `flow_match_mynet_wo_time_attention.yaml` | Ablation: no time, no attention     |
| SegResNet (flow)          | `flow_match_segresnet.yaml`             | MONAI SegResNet backbone              |
| SwinUNETR (flow)          | `flow_match_swinunetr.yaml`             | MONAI Swin UNETR backbone            |
| UMambaBot (flow)          | `flow_match_umamba.yaml`                | U-Mamba backbone                      |

### Regression Baselines

| Model         | Config                  |
|---------------|-------------------------|
| UNet          | `reg_unet.yaml`         |
| SegResNet     | `reg_segresnet.yaml`    |
| SwinUNETR     | `reg_swinunetr.yaml`    |
| UMambaBot     | `reg_umamba.yaml`       |

### Diffusion Baseline

| Model          | Config               |
|----------------|----------------------|
| DiffusionModel | `diffusion_net.yaml` |


## Training

All models are trained for 30 epochs with Adam optimizer (lr=2e-5) on 512×512 2D CT slices. Data augmentation includes random flips, rotations, and affine transforms.

### How to Train

Each model can be trained via its corresponding Jupyter notebook in `notebooks/`:

```python
# Example: train TimeResNet
from training.flow_matching import FLowTrainer
trainer = FLowTrainer('./configs/flow_match_mynet.yaml')
trainer.fit()
```

Training scripts:
- `training/flow_matching.py` — Flow matching trainer (all flow models)
- `training/reg_trainer.py` — Regression trainer (UNet, SegResNet, SwinUNETR, UMambaBot)
- `training/diffusion_trainer.py` — Diffusion model trainer
- `training/dataset.py` — Dataset loader with MONAI transforms

## Validation

Validation scripts evaluate trained models on test and holdout sets:

- **Flow matching**: Runs 6 ODE solvers (Euler 1/2/3 step, RK2, RK4, Midpoint) in both directions (contrast→native, native→contrast)
- **Regression**: Single forward pass, contrast→native only

### How to Run Validation

```bash
# Single flow matching model
python validation/validate_flow_models.py --gpu 0 --checkpoint <checkpoint_name>.pth

# All flow matching models (multi-GPU)
bash validation/run_validation.sh

# All regression models (multi-GPU)
bash validation/run_validation_regression.sh

# Aggregate results into summary JSON
python validation/validate_flow_models.py --aggregate
python validation/validate_regression_models.py --aggregate
```

## Training Logs

The `logs/` directory contains per-epoch loss values for all training runs. Each JSON file is a sequence of JSON lines with epoch number, train/test loss, and learning rate.

### Checkpoint-to-Model Mapping

| Checkpoint name                    | Model                     | Type           |
|------------------------------------|---------------------------|----------------|
| `run_lemon-shape-1_model`          | TimeResNet (ours)         | Flow Matching  |
| `run_misunderstood-cloud-7_model`  | TimeResNet w/o attention  | Flow Matching  |
| `run_still-sky-6_model`            | TimeResNet w/o time       | Flow Matching  |
| `run_polar-lion-8_model`           | TimeResNet w/o time+attn  | Flow Matching  |
| `run_vivid-lion-3_model`           | SegResNet                 | Flow Matching  |
| `run_royal-breeze-5_model`         | SwinUNETR                 | Flow Matching  |
| `run_volcanic-dragon-2_model`      | UMambaBot                 | Flow Matching  |
| `run_holographic-midichlorian-1_model` | DiffusionModel        | Diffusion      |
| `run_ancient-brook-1_model`        | UNet                      | Regression     |
| `run_sandy-moon-3_model`           | SegResNet                 | Regression     |
| `run_grievous-senate-4_model`      | SwinUNETR                 | Regression     |
| `run_ancient-bantha-5_model`       | UMambaBot                 | Regression     |

## Validation Results

Pre-computed validation results are provided in `results/`:
- `results/flow_matching/validation_results/` — Per-patient JSON files organized as `{ModelName}/{direction}/{dataset}/{solver}/{patient_id}.json`
- `results/regression/validation_results_reg/` — Per-patient JSON files organized as `{ModelName}/{direction}/{dataset}/{patient_id}.json`
- `summary.json` in each results directory contains aggregated mean±std metrics across patients
- `results/SMILE_results_test.json`, `results/SMILE_results_hold.json` - results of SMILE (SOTA) on test and hold datasets.

Each patient JSON contains: MAE (HU), PSNR (dB), SSIM, and inference time.

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- MONAI 1.3+
- dynamic-network-architectures (for UMambaBot)
- mamba-ssm (for UMambaBot, requires compatible CUDA)
- numpy, matplotlib, tqdm

## Figures

The `images/` directory contains pre-generated figures used in the paper:

- `flow_loss_plot.pdf` — Flow matching training curves (architecture comparison + ablation)
- `reg_loss_plot.pdf` — Regression baseline training curves
- `dif_loss_plot.pdf` — Diffusion model training curve
- `projection_montage.png` — Multi-model qualitative comparison (axial/sagittal/coronal projections + difference maps)
- `small_example.png` — Example CT translation result

These figures can be regenerated using the plotting notebooks:
- `notebooks/plot.ipynb` — Training loss curves and parameter counts
- `notebooks/visualize_results.ipynb` — Projection montage from pre-computed inference volumes
