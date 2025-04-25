# Phase 2: Conditional Diffusion Training & Evaluation

This directory contains Phase 2 scripts for:

1. **Labeling** 32×32 GoL patterns with life-type categories
2. **Training** a conditional diffusion model to learn embeddings for each category
3. **Evaluating** reproducibility by comparing dataset vs. generated proportions

Files:

- `label_training_data_32x32.py`
  - Simulates each `.npy` pattern and assigns one of: `died_out`, `still_life`, `oscillator_period_{n}`, `glider`, `spaceship`, or `others`.
  - Outputs `phase2_training_labels_32x32.csv`.

- `category_dataset.py`
  - PyTorch `Dataset` that reads `phase2_training_labels_32x32.csv` and returns `(tensor, label_idx)`.

- `train_conditional_diffusion.py`
  - Trains a **conditional** diffusion UNet (with `class_emb`) on the labeled data.
  - Hyperparameters match Phase 1 (`--timesteps 200`, etc.) and the provided command:
    ```bash
    python train_conditional_diffusion.py \
      --label_csv phase2_training_labels_32x32.csv \
      --model_ckpt checkpoints/model_5000.pt \
      --cf_prob 0.1 --weight_decay 1e-4 --dropout 0.0 \
      --epochs 250 --batch_size 32 --timesteps 200 --device cuda
    ```

- `evaluate_reproducibility.py`
  - Loads the trained conditional diffusion model and generates samples **per category**.
  - Classifies generated outputs using Phase 1 logic.
  - Computes percentages in dataset vs. generated for each category.
  - Ranks categories by `|delta%|` (reproducibility).
  - Writes `phase2_conditional_reproducibility.csv` and prints a summary.

Run each script in order for a full Phase 2 pipeline.
