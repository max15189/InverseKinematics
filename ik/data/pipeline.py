"""
Dataset generation, loading, and train/val/test splitting.

Workflow:
  1. generate_dataset_for_local_jacobian() or generate_dataset_for_hot_start()
  2. save_raw()        — save raw .npy files
  3. split_and_save()  — split into train / val / test and save per-split files
  4. load_split()      — load a single split back from disk

All functions that touch dq accept a hot_start: bool flag.
When hot_start=True, dq is ignored / not saved / not loaded.
"""
import sys
sys.path.insert(0, r'c:\Master\projects\IK')

import numpy as np
from sklearn.model_selection import train_test_split

from ik.kinematics.fk import FK, JOINT_LIMITS
#from ik.kinematics.fk import FK,JOINT_LIMITS
# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def sample_q(n: int) -> np.ndarray:
    """
    Sample n joint configurations uniformly within joint limits.

    Returns:
        (n, 6) array of joint angles in radians.
    """
    lo = JOINT_LIMITS[:, 0]
    hi = JOINT_LIMITS[:, 1]
    return lo + np.random.rand(n, 6) * (hi - lo)


def perturb_q(q: np.ndarray, delta: float = 0.05, n_neighbours: int = 5) -> np.ndarray:
    """
    Generate n_neighbours nearby configs by adding uniform noise in
    [-delta, +delta] per joint, clamped to joint limits.

    Args:
        q:            (6,) base joint config.
        delta:        max perturbation per joint in radians (~2.8° at 0.05).
        n_neighbours: number of perturbed configs to generate.

    Returns:
        (n_neighbours, 6) array.
    """
    lo = JOINT_LIMITS[:, 0]
    hi = JOINT_LIMITS[:, 1]
    noise = (np.random.rand(n_neighbours, 6) * 2 - 1) * delta
    return np.clip(q + noise, lo, hi)


def generate_dataset_for_hot_start(n: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate (q, xd) pairs for training a hot-start predictor.
    The MLP learns to map xd → q directly (no delta).

    Returns:
        q_all:  (n, 6) joint configs
        xd_all: (n, 3) end-effector positions
    """
    q_all  = sample_q(n)
    xd_all = np.zeros((n, 3))
    for i, q in enumerate(q_all):
        if i % 5000 == 0:
            print(f"  generating... {i}/{n}")
        xd_all[i] = FK(q)[0:3, 3]
    return q_all, xd_all


def generate_dataset_for_local_jacobian(
    n_base: int = 50_000,
    n_neighbours: int = 5,
    delta: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate (q1, x_d, dq) tuples for learning a local Jacobian.

    For each base config q1:
      - sample n_neighbours nearby configs q2
      - compute x_d = FK(q2)[0:3, 3]
      - label   dq  = q2 - q1

    Returns:
        q1_all: (N, 6)  current joint configs
        xd_all: (N, 3)  target end-effector positions
        dq_all: (N, 6)  joint deltas to apply
      where N = n_base * n_neighbours.
    """
    N = n_base * n_neighbours
    q1_all = np.zeros((N, 6))
    xd_all = np.zeros((N, 3))
    dq_all = np.zeros((N, 6))

    base_configs = sample_q(n_base)
    idx = 0
    for i, q1 in enumerate(base_configs):
        if i % 5000 == 0:
            print(f"  generating... {i}/{n_base}")
        for q2 in perturb_q(q1, delta=delta, n_neighbours=n_neighbours):
            x_d          = FK(q2)[0:3, 3]
            q1_all[idx]  = q1
            xd_all[idx]  = x_d
            dq_all[idx]  = q2 - q1
            idx += 1

    return q1_all, xd_all, dq_all


def validate_dataset(
    q1_all: np.ndarray,
    xd_all: np.ndarray,
    dq_all: np.ndarray = None,
    n_check: int = 1000,
    hot_start: bool = False,
) -> None:
    """
    Spot-check dataset correctness.

    hot_start=False: verifies FK(q1 + dq) ≈ x_d
    hot_start=True:  verifies FK(q1)      ≈ x_d
    """
    indices = np.random.choice(len(q1_all), n_check, replace=False)

    if hot_start:
        errors = [
            np.linalg.norm(FK(q1_all[i])[0:3, 3] - xd_all[i])
            for i in indices
        ]
    else:
        errors = [
            np.linalg.norm(FK(q1_all[i] + dq_all[i])[0:3, 3] - xd_all[i])
            for i in indices
        ]

    errors = np.array(errors)
    print(f"\nDataset validation ({n_check} samples):")
    print(f"  mean position error: {errors.mean():.6e} m")
    print(f"  max  position error: {errors.max():.6e} m")
    print("  (should be ~0 — any error is a bug in FK or data gen)")


# ---------------------------------------------------------------------------
# Saving & loading
# ---------------------------------------------------------------------------

def save_raw(
    save_dir: str,
    q1: np.ndarray,
    xd: np.ndarray,
    dq: np.ndarray = None,
    hot_start: bool = False,
) -> None:
    """Save raw arrays to save_dir. dq is skipped when hot_start=True."""
    np.save(f"{save_dir}/q1.npy", q1)
    np.save(f"{save_dir}/xd.npy", xd)
    if not hot_start:
        np.save(f"{save_dir}/dq.npy", dq)
        print(f"Saved q1.npy, xd.npy, dq.npy → {save_dir}")
    else:
        print(f"Saved q1.npy, xd.npy → {save_dir}")


def load_raw(
    save_dir: str,
    hot_start: bool = False,
) -> tuple:
    """
    Load raw arrays from save_dir.

    Returns:
        hot_start=False: (q1, xd, dq)
        hot_start=True:  (q1, xd)
    """
    q1 = np.load(f"{save_dir}/q1.npy")
    xd = np.load(f"{save_dir}/xd.npy")
    if hot_start:
        return q1, xd
    dq = np.load(f"{save_dir}/dq.npy")
    return q1, xd, dq


def split_and_save(
    save_dir: str,
    test_size: float = 0.20,
    val_fraction: float = 0.50,
    random_state: int = 42,
    hot_start: bool = False,
) -> None:
    """
    Load raw arrays, split 80 / 10 / 10 (train / val / test), and save
    per-split files to save_dir. dq files are skipped when hot_start=True.
    """
    if hot_start:
        q1, xd = load_raw(save_dir, hot_start=True)
        dq = None
    else:
        q1, xd, dq = load_raw(save_dir, hot_start=False)

    indices = np.arange(len(q1))
    idx_train, idx_temp = train_test_split(indices, test_size=test_size, random_state=random_state)
    idx_val,   idx_test = train_test_split(idx_temp, test_size=1 - val_fraction, random_state=random_state)

    splits = {"train": idx_train, "val": idx_val, "test": idx_test}
    for name, idx in splits.items():
        np.save(f"{save_dir}/q1_{name}.npy", q1[idx])
        np.save(f"{save_dir}/xd_{name}.npy", xd[idx])
        if not hot_start:
            np.save(f"{save_dir}/dq_{name}.npy", dq[idx])
        print(f"{name:>5}: {len(idx):>7,} samples")

    print(f"\ntotal: {len(q1):,}")


def load_split(
    save_dir: str,
    split: str,
    hot_start: bool = False,
) -> tuple:
    """
    Load one split (train / val / test) from save_dir.

    Returns:
        hot_start=False: (q1, xd, dq) as float32 arrays
        hot_start=True:  (q1, xd)     as float32 arrays
    """
    q1 = np.load(f"{save_dir}/q1_{split}.npy").astype(np.float32)
    xd = np.load(f"{save_dir}/xd_{split}.npy").astype(np.float32)
    if hot_start:
        return q1, xd
    dq = np.load(f"{save_dir}/dq_{split}.npy").astype(np.float32)
    return q1, xd, dq


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    SAVE_DIR  = "G:/My Drive/inverse_kinematics/hot_start"
    HOT_START = True   

    os.makedirs(SAVE_DIR, exist_ok=True)
    np.random.seed(42)

    if HOT_START:
        print("Generating hot-start dataset...")
        q1, xd = generate_dataset_for_hot_start(n=100_000)
        print(f"\nDataset size: {len(q1):,} samples")
        validate_dataset(q1, xd, hot_start=True)
        save_raw(SAVE_DIR, q1, xd, hot_start=True)
    else:
        print("Generating local-Jacobian dataset...")
        q1, xd, dq = generate_dataset_for_local_jacobian(n_base=50_000, n_neighbours=5, delta=0.05)
        print(f"\nDataset size: {len(q1):,} samples")
        validate_dataset(q1, xd, dq, hot_start=False)
        save_raw(SAVE_DIR, q1, xd, dq, hot_start=False)

    print("\nSplitting dataset...")
    split_and_save(SAVE_DIR, hot_start=HOT_START)
