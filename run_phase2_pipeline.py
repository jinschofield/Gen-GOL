#!/usr/bin/env python3
"""
Run full Phase 2 pipeline with progress prints:
 1. Labeling
 2. Generating balanced dataset
 3. Training conditional diffusion
 4. Evaluating reproducibility
"""
import subprocess
import sys

def run(cmd, stage):
    print(f"\n=== Stage: {stage} ===")
    print(f"Command: {' '.join(cmd)}")
    subprocess.check_call(cmd)

if __name__ == '__main__':
    # Adjust paths as needed
    base = os.getcwd()
    label_csv = 'phase_2_conditional_diffusion/phase2_training_labels_32x32.csv'
    balanced_csv = 'phase_2_conditional_diffusion/balanced_labels_32x32.csv'
    model_ckpt = 'phase2_results/cond_model_final.pt'

    # Stage 1: Labeling
    run([sys.executable, '-m', 'phase_2_conditional_diffusion.label_training_data_32x32'], 'Label Training Data')

    # Stage 2: Generate Balanced Dataset
    run([sys.executable, '-m', 'phase_2_conditional_diffusion.generate_balanced_dataset',
         f'--target_count', '1000'], 'Generate Balanced Dataset')

    # Stage 3: Train Conditional Diffusion
    train_cmd = [sys.executable, '-m', 'phase_2_conditional_diffusion.train_conditional_diffusion',
                 f'--label_csv', balanced_csv,
                 '--epochs', '250', '--batch_size', '32', '--timesteps', '200', '--device', 'cuda']
    run(train_cmd, 'Train Conditional Diffusion')

    # Stage 4: Evaluate Reproducibility
    eval_cmd = [sys.executable, '-m', 'phase_2_conditional_diffusion.evaluate_reproducibility',
                '--model_ckpt', model_ckpt,
                f'--label_csv', balanced_csv,
                '--num_gen', '1000', '--timesteps', '200', '--schedule', 'cosine',
                '--guidance_scale', '1.0', '--device', 'cuda',
                '--output_csv', 'phase2_results/phase2_cond_balanced_reproducibility.csv']
    run(eval_cmd, 'Evaluate Reproducibility')

    print("\nPhase 2 pipeline completed successfully.")
