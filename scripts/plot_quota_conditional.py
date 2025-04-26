#!/usr/bin/env python3
"""
Plot conditional types for quota model only.
"""
import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import numpy as np
import torch
import matplotlib.pyplot as plt

from generate_report import UNet, Diffusion, sample_model, cats_types

# CONFIGURATION
CKPT        = 'finished_models/model_final_quota.pt'
DEVICE      = 'cuda'
TIMESTEPS   = 200
NUM_SAMPLES = 1000

# Load model and diffusion
model = UNet(dropout=0.0, num_classes=5).to(DEVICE)
state = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(state)
model.eval()
diff = Diffusion(timesteps=TIMESTEPS,
                 device=DEVICE,
                 schedule='cosine',
                 guidance_scale=1.0)

# Sample conditional percentages
cond_types = [('still_life',1), ('oscillator_period_2',2), ('others',3)]
cond_pcts = {}
for name, cond in cond_types:
    print(f"[plot_quota_conditional] Sampling for {name}...")
    _, pct = sample_model(model, diff, DEVICE, NUM_SAMPLES, TIMESTEPS, cond, types=True)
    cond_pcts[name] = pct

# Plot conditional types
x = np.arange(len(cats_types))
width = 0.2
fig, ax = plt.subplots(figsize=(6,4))
for i, (name, _) in enumerate(cond_types):
    vals = [cond_pcts[name][c] for c in cats_types]
    ax.bar(x + i*width, vals, width, label=name)
ax.set_xticks(x + width)
ax.set_xticklabels(cats_types, rotation=45)
ax.legend()
fig.tight_layout()
fig.savefig('quota_conditional_types.png')
print("[plot_quota_conditional] Saved quota_conditional_types.png")
