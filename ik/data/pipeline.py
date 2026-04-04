"""
Dataset generation, loading, and train/val/test splitting.

Workflow:
  1. generate_euclidean_q_targets() — generate q_init/q_dest pairs
  2. save_raw()        — save raw .npy files
  3. split_and_save()  — split into train / val / test and save per-split files
  4. load_split()      — load a single split back from disk
"""
import numpy as np
from sklearn.model_selection import train_test_split

from ik.kinematics.fk import JOINT_LIMITS

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_euclidean_q_targets(q_init):
    """
    Generates 72 pairs per q_init across 4 joint-space Euclidean distance tiers.
    Far tiers are oversampled to ensure representation despite joint limit clipping.
    Each (q_init, q_dest) pair is also included reversed as (q_dest, q_init) —
    L2 distance is symmetric so tier membership is preserved and the dataset doubles.

    Args:
        q_init: np.array of shape (n, 6) in radians
    Returns:
        q_init_all: (72*n, 6)
        q_dest_all: (72*n, 6)
    """
    n = q_init.shape[0]

    low_lim, high_lim = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]

    # (deg_min, deg_max, n_samples)
    # Far tiers get more samples to compensate for joint-limit clipping
    tiers_cfg = [
        (1,   3,   6),   # close
        (5,  15,   8),   # medium
        (15, 60,  10),   # far     (extended from original 30)
        (60, 120, 12),   # very far
    ]
    total_samples = sum(n for _, _, n in tiers_cfg)  # 36

    # Expand q_init to (n, total_samples, 6)
    q_expanded = np.repeat(q_init[:, np.newaxis, :], total_samples, axis=1)

    # Random directions — uniform on unit sphere in 6D
    directions = np.random.randn(n, total_samples, 6)
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)

    # Random magnitudes per tier, concatenated in order
    mags_list = []
    for deg_min, deg_max, n_samples in tiers_cfg:
        low  = np.deg2rad(deg_min)
        high = np.deg2rad(deg_max)
        mags_list.append(np.random.uniform(low, high, size=(n, n_samples)))

    magnitudes = np.concatenate(mags_list, axis=1)[:, :, np.newaxis]  # (n, 36, 1)

    q_dest = q_expanded + (directions * magnitudes)
    q_dest = np.clip(q_dest, low_lim, high_lim)

    q_init_flat = q_expanded.reshape(-1, 6)
    q_dest_flat = q_dest.reshape(-1, 6)

    # Symmetric augmentation: also treat each q_dest as a starting config.
    # L2 distance is symmetric so tier membership is preserved.
    q_init_all = np.concatenate([q_init_flat, q_dest_flat], axis=0)
    q_dest_all = np.concatenate([q_dest_flat, q_init_flat], axis=0)

    return q_init_all, q_dest_all


# ---------------------------------------------------------------------------
# Save / split / load
# ---------------------------------------------------------------------------

def save_raw(
    save_dir: str,
    q_init: np.ndarray,
    q_target: np.ndarray,
    P_target: np.ndarray,
    R6_target: np.ndarray,
) -> None:
    """Saves the raw generated data as .npy files."""
    np.save(f"{save_dir}/q_init.npy",    q_init.astype(np.float32))
    np.save(f"{save_dir}/q_target.npy",  q_target.astype(np.float32))
    np.save(f"{save_dir}/P_target.npy",  P_target.astype(np.float32))
    np.save(f"{save_dir}/R6_target.npy", R6_target.astype(np.float32))
    print(f"Saved raw data to {save_dir}")


def split_and_save(
    save_dir: str,
    test_size: float = 0.20,
    val_fraction: float = 0.50,
    random_state: int = 42,
) -> None:
    """Loads raw arrays, splits into train/val/test, and saves with split suffixes."""
    q_init    = np.load(f"{save_dir}/q_init.npy")
    q_target  = np.load(f"{save_dir}/q_target.npy")
    P_target  = np.load(f"{save_dir}/P_target.npy")
    R6_target = np.load(f"{save_dir}/R6_target.npy")

    indices = np.arange(len(q_init))
    idx_train, idx_temp = train_test_split(indices, test_size=test_size, random_state=random_state)
    idx_val, idx_test   = train_test_split(idx_temp, test_size=1 - val_fraction, random_state=random_state)

    splits = {"train": idx_train, "val": idx_val, "test": idx_test}
    for name, idx in splits.items():
        np.save(f"{save_dir}/q_init_{name}.npy",    q_init[idx])
        np.save(f"{save_dir}/q_target_{name}.npy",  q_target[idx])
        np.save(f"{save_dir}/P_target_{name}.npy",  P_target[idx])
        np.save(f"{save_dir}/R6_target_{name}.npy", R6_target[idx])
        print(f"{name:>5}: {len(idx):>7,} samples")


def load_split(save_dir: str, split: str) -> tuple:
    """Loads one specific split. Returns all arrays as float32."""
    q_init    = np.load(f"{save_dir}/q_init_{split}.npy").astype(np.float32)
    q_target  = np.load(f"{save_dir}/q_target_{split}.npy").astype(np.float32)
    P_target  = np.load(f"{save_dir}/P_target_{split}.npy").astype(np.float32)
    R6_target = np.load(f"{save_dir}/R6_target_{split}.npy").astype(np.float32)
    return q_init, q_target, P_target, R6_target
