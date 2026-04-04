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

from ik.kinematics.fk import FK, JOINT_LIMITS
from ik.solver.iterative_ik import nr_ik

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_euclidean_q_targets(q_init):
    """
    Generates 30 targets per q_init based on Euclidean distance tiers in joint space.
    Optimized for ML training sets with mixed transformation scales.

    Args:
        q_init: np.array of shape (n, 6) in radians
    Returns:
        q_init_expanded: (30*n, 6)
        q_dest: (30*n, 6)
    """
    n = q_init.shape[0]
    samples_per_tier = 10
    total_samples = 30

    low_lim, high_lim = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]

    # Define Euclidean Distance Tiers (Degrees -> Radians)
    tiers_deg = np.array([
        [1, 3],    # Close
        [5, 15],   # Further
        [15, 30]   # Far
    ])
    tiers_rad = np.deg2rad(tiers_deg)

    # Expand q_init to (n, 30, 6)
    q_expanded = np.repeat(q_init[:, np.newaxis, :], total_samples, axis=1)

    # Random directions (uniform on unit sphere in 6D)
    directions = np.random.randn(n, total_samples, 6)
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)

    # Random magnitudes per tier
    mags_list = []
    for low, high in tiers_rad:
        mags_list.append(np.random.uniform(low, high, size=(n, samples_per_tier)))

    magnitudes = np.concatenate(mags_list, axis=1)[:, :, np.newaxis]

    q_dest = q_expanded + (directions * magnitudes)
    q_dest = np.clip(q_dest, low_lim, high_lim)

    return q_expanded.reshape(-1, 6), q_dest.reshape(-1, 6)

from tqdm import tqdm

def generate_task_space_targets(q_init, n_per_q: int = 36, seed: int | None = None):
    """
    Generates targets by sampling in task space around each q_init pose.

    For each q_init:
      1. Compute T_init = FK(q_init)
      2. Perturb position by a random vector (sampled within a sphere)
         and orientation by a random axis-angle rotation.
      3. Run NR IK from q_init to find q_target for the perturbed pose.
      4. Keep only converged pairs.

    Tiers are defined in (pos metres, rot degrees) — directly meaningful
    for task-space difficulty, unlike joint-space L2 norms.
    Far tiers are oversampled to compensate for higher NR failure rates.

    Args:
        q_init:   (n, 6) initial joint configurations in radians
        n_per_q:  total target samples per q_init across all tiers (default 36)
        seed:     optional random seed

    Returns:
        q_inits:   (m, 6) — only converged pairs
        q_targets: (m, 6)
    """
    if seed is not None:
        np.random.seed(seed)

    # (pos_min_m, pos_max_m, rot_min_deg, rot_max_deg, n_samples)
    # Far tiers get more attempts to compensate for NR failures
    tiers = [
        (0.001, 0.02,  1,   5,  6),    # close    — easy, fewest attempts
        (0.02,  0.10,  5,  20,  8),    # medium
        (0.10,  0.30, 20,  60, 10),    # far
        (0.30,  0.50, 60, 120, 12),    # very far — hardest, most attempts
    ]

    all_q_init, all_q_target = [], []

    for q0 in tqdm(q_init, desc="generating"):
        T0 = FK(q0)
        p0 = T0[:3, 3]
        R0 = T0[:3, :3]

        for p_min, p_max, r_min_deg, r_max_deg, n in tiers:
            r_min = np.deg2rad(r_min_deg)
            r_max = np.deg2rad(r_max_deg)

            for _ in range(n):
                # --- position perturbation: uniform inside spherical shell ---
                d   = np.random.randn(3)
                d  /= np.linalg.norm(d)
                r   = np.random.uniform(p_min, p_max)
                p_t = p0 + r * d

                # --- orientation perturbation: random axis-angle ---
                w     = np.random.randn(3)
                w    /= np.linalg.norm(w)
                theta = np.random.uniform(r_min, r_max)
                W     = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
                R_per = np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * W @ W
                R_t   = R_per @ R0

                T_target        = np.eye(4)
                T_target[:3, :3] = R_t
                T_target[:3,  3] = p_t

                q_sol, converged, _ = nr_ik(T_target, q0)
                if converged:
                    all_q_init.append(q0)
                    all_q_target.append(q_sol)

    return np.array(all_q_init, dtype=np.float32), np.array(all_q_target, dtype=np.float32)


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
