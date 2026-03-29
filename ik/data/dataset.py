"""
PyTorch Dataset for the IK dataset.
Supports both local-Jacobian mode (q1, xd → dq) and hot-start mode (xd → q).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from ik.data.pipeline import load_split


class IKDataset(Dataset):
    """
    Args:
        split:      one of 'train', 'val', 'test'
        save_dir:   directory where the split .npy files live
        hot_start:  if True, input=xd (3,), target=q (6,) — no dq
                    if False, input=[q1, xd] (9,), target=dq (6,)
        scaler_X:   [mean, std] tensors for X; computed from data if None
        scaler_Y:   [mean, std] tensors for Y; computed from data if None
    """

    def __init__(
        self,

        split: str,
        save_dir: str,
        hot_start: bool = False,
        scaler_X: list | None = None,
        scaler_Y: list | None = None,
    ):
        self.hot_start = hot_start

        if hot_start:
            q1_np, xd_np = load_split(save_dir, split, hot_start=True)
            X_raw = torch.from_numpy(xd_np)          # input  = xd (3,)
            from ik.kinematics.fk import FK_batch_full
            
            Y_raw = torch.from_numpy(q1_np)          # target = q  (6,)
            angular_Raw = FK_batch_full(Y_raw)[:, :2, 0:3].reshape(len(Y_raw), -1)  # (B, 6)
            X_raw = torch.cat([X_raw, angular_Raw], dim=1)                          # (B, 9)
        else:
            q1_np, xd_np, dq_np = load_split(save_dir, split, hot_start=False)
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
