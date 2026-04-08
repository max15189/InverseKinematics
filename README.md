# Inverse Kinematics with Neural Network warm start — ViperX-300

A learning project inspired by the Modern Robotics course and the desire to improve my skills by working on real world problems

The idea: train an MLP to predict a good initial joint configuration, then let Newton-Raphson refine it to full precision. The MLP alone is fast but approximate; NR alone is precise but sensitive to the starting point. Together they converge faster — especially when the target is far from the current configuration.

---

## How it works

1. **Input** — 15-dimensional vector: position of the end-effector (3), first two rows of the rotation matrix (6, the third row provide no extra information) and  the current joint configuration (6).
2. **MLP** — one forward pass predicts a joint configuration close to the target.
3. **Newton-Raphson** — starts from the MLP prediction and refines until convergence.

For full details on the dataset design, loss function, and benchmark results see learned_warmstart_for_nr_ik_viperx300.docx


---

## Quick start

### Install dependencies

```bash
pip install torch numpy modern-robotics scikit-learn tqdm matplotlib
```

### Clone and run

```bash
git clone https://github.com/max15189/InverseKinematics.git
cd InverseKinematics
```

### Solve IK in Python

```python
from ik.inference import IKSolver
import numpy as np

solver = IKSolver()  # loads model and normalisation stats from model/

# Single solve — give current joints (rad) and desired 4x4 transform
result = solver.solve(q_current, T_target)
print(result["q"])          # (6,) solution in radians
print(result["converged"])  # True/False
print(result["iterations"]) # NR iterations taken

# Batch solve — (N, 6) and (N, 4, 4)
results = solver.solve_batch(q_currents, T_targets)
```

The solver warns automatically if any joint is outside the ViperX-300 limits.

---

## Project structure

```
ik/
  viperx300.py      — robot parameters (joint limits, screw axes, home config)
  kinematics/fk.py  — forward kinematics (NumPy + differentiable PyTorch)
  data/
    pipeline.py     — dataset generation
    dataset.py      — PyTorch Dataset
  model/
    mlp.py          — MLP architecture
    training.py     — training loop with combined joint + task-space loss
  solver/
    nr_ik.py — Newton-Raphson IK (Modern Robotics with added iteration in the return)
  inference.py      — user-facing IKSolver API

notebooks/
  DataGenerator.ipynb      — generate and save the dataset
  train_ik.ipynb           — train the MLP
  evaluate_model.ipynb     — evaluate on train / val / test splits
  benchmark_mlp_vs_nr.ipynb — 4 benchmarks comparing warm vs cold start
  test_inference.ipynb     — test the IKSolver API end-to-end

model/
  model.pt        — trained weights
  MinMax_X.npy    — input normalisation stats
  MinMax_Y.npy    — output normalisation stats

config.json       — all paths and hyperparameters
```

---

## Configuration

All paths and hyperparameters are in `config.json`. Edit it to point to your dataset and model locations before running any notebook.

```json
{
    "SAVE_DATA":  "dataset",
    "SAVE_MODEL": "model/model.pt",
    "EPOCHS": 200,
    "LR": 1e-3,
    ...
}
```

---

## Robot

ViperX-300 6-DOF arm by Trossen Robotics. All kinematic parameters are in `ik/viperx300.py`.  
Spec: https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html

