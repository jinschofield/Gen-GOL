# Diffusion-Based Generative Model for Conway’s Game of Life

This project implements an **unconditional denoising diffusion probabilistic model (DDPM)** to generate novel Conway’s Game of Life patterns (oscillators, spaceships, still lifes). It includes:

- **Data preparation**: load & augment binary GoL patterns
- **Model**: U-Net + diffusion noise scheduler (Gaussian forward process)
- **Training**: PyTorch training loop with MSE loss on noise prediction
- **Evaluation**: sample patterns, simulate GoL dynamics, compute oscillator/novelty metrics

## Requirements
See `requirements.txt`.

## Project Structure
```
/Users/jinschofield/Downloads/gen-gol
├── data/
│   └── dataset.py        # dataset loader for .npy patterns
├── models/
│   ├── unet.py           # U-Net architecture
│   └── diffusion.py      # Diffusion helper (noise schedule, sampling)
├── train.py              # Training script
├── evaluate.py           # Sampling & evaluation script
├── utils/
│   ├── gol_simulator.py  # GoL simulation routines
│   └── metrics.py        # Pattern validity & novelty metrics
├── requirements.txt      # Python dependencies
└── README.md             # Project overview
```
