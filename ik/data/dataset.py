"""
PyTorch Dataset for the IK local-Jacobian dataset.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from ik.data.pipeline import load_split


class IKDataset(Dataset):
    """
    Dataset of (q1, x_d, dq) tuples for local inverse-kinematics learning.

    Input  X: concatenation of q1 (6) and x_d (3), z-score normalised → (9,)
    Target Y: dq (6), z-score normalised → (6,)

    Raw q1 and xd tensors are also stored for use in task-space loss
    computation during training.

    Args:
        split:     one of 'train', 'val', 'test'
        save_dir:  directory where the split .npy files live
        scaler_X:  [mean, std] tensors for X; computed from data if None
        scaler_Y:  [mean, std] tensors for Y; computed from data if None
    """

    def __init__(
        self,
        split: str,
        save_dir: str,
        scaler_X: list | None = None,
        scaler_Y: list | None = None,
    ):
        q1_np, xd_np, dq_np = load_split(save_dir, split)

        X_raw = torch.from_numpy(np.concatenate([q1_np, xd_np], axis=1))
        Y_raw = torch.from_numpy(dq_np)

        if scaler_X is None:
            self.scaler_X = [X_raw.mean(dim=0), X_raw.std(dim=0)]
            self.scaler_Y = [Y_raw.mean(dim=0), Y_raw.std(dim=0)]
        else:
            self.scaler_X = scaler_X
            self.scaler_Y = scaler_Y

        self.X  = (X_raw - self.scaler_X[0]) / self.scaler_X[1]
        self.Y  = (Y_raw - self.scaler_Y[0]) / self.scaler_Y[1]
        self.q1 = torch.from_numpy(q1_np)
        self.xd = torch.from_numpy(xd_np)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.q1[idx], self.xd[idx]
