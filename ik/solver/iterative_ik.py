"""
Iterative inverse-kinematics solver.

Uses the trained MLP to iteratively refine a joint configuration
until the end-effector reaches the target position within a threshold.
"""

import numpy as np
import torch
import torch.nn as nn

from ik.kinematics.fk import FK_batch


def iterative_ik(
    model: nn.Module,
    q: torch.Tensor,
    xd: torch.Tensor,
    scaler_X: list,
    scaler_Y: list,
    device,
    threshold: float = 2e-3,
    max_iter: int = 20,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, list[float], bool]:
    """
    Iteratively apply the MLP to drive the end-effector towards xd.

    Args:
        model:     trained MLP (in eval mode)
        q:         (6,) initial joint configuration in radians
        xd:        (3,) target end-effector position in metres
        scaler_X:  [X_mean, X_std] tensors used during training
        scaler_Y:  [Y_mean, Y_std] tensors used during training
        device:    torch device
        threshold: convergence criterion in metres (default 2 mm)
        max_iter:  maximum number of iterations
        alpha:     step size / damping factor in (0, 1]

    Returns:
        q:             (6,) final joint configuration
        error_history: list of end-effector errors per iteration (metres)
        converged:     True if threshold was reached
    """
    X_mean, X_std = scaler_X[0].to(device), scaler_X[1].to(device)
    Y_mean, Y_std = scaler_Y[0].to(device), scaler_Y[1].to(device)

    q  = q.to(device)
    xd = xd.to(device)
    error_history: list[float] = []

    model.eval()
    for _ in range(max_iter):
        inp        = torch.cat([q, xd]).unsqueeze(0)          # (1, 9)
        inp_scaled = (inp - X_mean) / X_std

        with torch.no_grad():
            dq = model(inp_scaled) * Y_std + Y_mean           # (1, 6)

        q = q + alpha * dq.squeeze(0)

        error = torch.norm(xd - FK_batch(q.unsqueeze(0)).squeeze(0)).item()
        error_history.append(error)

        if error < threshold:
            return q, error_history, True

    return q, error_history, False
