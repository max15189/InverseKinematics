"""
Forward Kinematics for ViperX-300 6-DOF arm.
Provides both a NumPy implementation (FK) and a batched PyTorch
implementation (FK_batch) that supports autograd for task-space losses.

Robot spec: https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html
"""

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Robot constants
# ---------------------------------------------------------------------------

JOINT_LIMITS = np.deg2rad([
    [-180,  180],   # waist
    [-101,  101],   # shoulder
    [-101,   92],   # elbow
    [-180,  180],   # forearm roll
    [-107,  130],   # wrist angle
    [-180,  180],   # wrist rotate
])  # shape (6, 2)

# Screw axes — each ROW is [w; v] for one joint; transposed so each COLUMN
# is the screw axis for joint i (space-frame POE convention).
_Slist_np = np.array([
    [0,  0,  1,  0,        0,       0      ],  # joint 1 (waist)
    [0,  1,  0, -0.12705,  0,       0      ],  # joint 2 (shoulder)
    [0,  1,  0, -0.42705,  0,       0.05955],  # joint 3 (elbow)
    [1,  0,  0,  0,        0.42705, 0      ],  # joint 4 (forearm roll)
    [0,  1,  0, -0.42705,  0,       0.35955],  # joint 5 (wrist angle)
    [1,  0,  0,  0,        0.42705, 0      ],  # joint 6 (wrist rotate)
]).T  # shape (6, 6)

_M_HOME_np = np.array([
    [1, 0, 0, 0.536494],
    [0, 1, 0, 0       ],
    [0, 0, 1, 0.42705 ],
    [0, 0, 0, 1       ],
], dtype=float)

# PyTorch versions (created once; moved to device inside FK_batch)
_Slist_torch  = torch.tensor(_Slist_np,  dtype=torch.float32)
_M_HOME_torch = torch.tensor(_M_HOME_np, dtype=torch.float32)


# ---------------------------------------------------------------------------
# NumPy FK
# ---------------------------------------------------------------------------

def skew3(w: np.ndarray) -> np.ndarray:
    """3-vector → 3×3 skew-symmetric matrix."""
    w = w.flatten()
    return np.array([
        [ 0,    -w[2],  w[1]],
        [ w[2],  0,    -w[0]],
        [-w[1],  w[0],  0   ],
    ])


def vec_to_se3(S: np.ndarray) -> np.ndarray:
    """6-vector screw axis S=[w;v] → 4×4 se(3) matrix."""
    S = S.flatten()
    se3 = np.zeros((4, 4))
    se3[0:3, 0:3] = skew3(S[0:3])
    se3[0:3, 3]   = S[3:6]
    return se3


def se3_to_SE3(S: np.ndarray, theta: float) -> np.ndarray:
    """
    Exponential map for a revolute joint (||w|| = 1).
    Returns the 4×4 rigid-body transform in SE(3).
    """
    se3 = vec_to_se3(S)
    W   = se3[0:3, 0:3]
    v   = se3[0:3, 3]

    R = np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * (W @ W)
    G = np.eye(3) * theta + (1 - np.cos(theta)) * W + (theta - np.sin(theta)) * (W @ W)

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3]   = G @ v
    return T


def FK(q) -> np.ndarray:
    """
    Forward kinematics via Product of Exponentials (space frame).

    Args:
        q: array-like of 6 joint angles in radians.

    Returns:
        4×4 homogeneous transform from base to end-effector.
    """
    q = np.array(q).flatten()
    assert len(q) == 6, "Expected 6 joint angles"

    T = _M_HOME_np.copy()
    for i in range(5, -1, -1):
        T = se3_to_SE3(_Slist_np[:, i], q[i]) @ T
    return T


# ---------------------------------------------------------------------------
# PyTorch batched FK  (supports autograd)
# ---------------------------------------------------------------------------

def _skew3_batch(w: torch.Tensor) -> torch.Tensor:
    """w: (B, 3) → (B, 3, 3) skew-symmetric matrices."""
    B = w.shape[0]
    S = torch.zeros(B, 3, 3, device=w.device)
    S[:, 0, 1] = -w[:, 2];  S[:, 0, 2] =  w[:, 1]
    S[:, 1, 0] =  w[:, 2];  S[:, 1, 2] = -w[:, 0]
    S[:, 2, 0] = -w[:, 1];  S[:, 2, 1] =  w[:, 0]
    return S


def _se3_to_SE3_batch(S: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Args:
        S:     (6,)  screw axis [w; v]
        theta: (B,)  joint angles

    Returns:
        T: (B, 4, 4)
    """
    B  = theta.shape[0]
    w  = S[:3]
    v  = S[3:]

    W = _skew3_batch(w.unsqueeze(0).expand(B, 3))   # (B, 3, 3)

    sin_t = torch.sin(theta).view(B, 1, 1)
    cos_t = torch.cos(theta).view(B, 1, 1)
    t     = theta.view(B, 1, 1)
    I     = torch.eye(3, device=theta.device).unsqueeze(0)

    R = I + sin_t * W + (1 - cos_t) * (W @ W)
    G = I * t + (1 - cos_t) * W + (t - sin_t) * (W @ W)

    p = (G @ v.view(1, 3, 1).expand(B, 3, 1)).squeeze(-1)  # (B, 3)

    T = torch.eye(4, device=theta.device).unsqueeze(0).expand(B, 4, 4).clone()
    T[:, :3, :3] = R
    T[:, :3,  3] = p
    return T


def FK_batch(q: torch.Tensor) -> torch.Tensor:
    """
    Batched differentiable forward kinematics.

    Args:
        q: (B, 6) joint angles in radians.

    Returns:
        end-effector positions: (B, 3)
    """
    B   = q.shape[0]
    dev = q.device
    Sl  = _Slist_torch.to(dev)
    M   = _M_HOME_torch.to(dev)

    T = M.unsqueeze(0).expand(B, 4, 4).clone()
    for i in range(5, -1, -1):
        T = _se3_to_SE3_batch(Sl[:, i], q[:, i]) @ T

    return T[:, :3, 3]   # (B, 3)
