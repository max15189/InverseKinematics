"""
Newton-Raphson IK solver using the Modern Robotics library.

Uses the space-frame Product-of-Exponentials formulation consistent
with the ViperX-300 FK defined in ik.kinematics.fk.
"""

import numpy as np
import modern_robotics as mr

from ik.kinematics.fk import _Slist_np as Slist, _M_HOME_np as M_home


def nr_ik(
    T_target: np.ndarray,
    q_init: np.ndarray,
    eomg: float = 0.01,
    ev: float = 1e-3,
    max_iter: int = 50,
) -> tuple[np.ndarray, bool, int]:
    """
    Damping-free Newton-Raphson IK via the Modern Robotics space Jacobian.

    Args:
        T_target:  (4, 4) desired end-effector transform
        q_init:    (6,)   initial joint configuration in radians
        eomg:      orientation error tolerance (rad)
        ev:        position error tolerance (m)
        max_iter:  maximum NR iterations

    Returns:
        q:          (6,) solution joint configuration
        converged:  True if both eomg and ev tolerances were met
        iterations: number of iterations taken
    """
    thetalist = np.array(q_init, dtype=float).copy()
    i   = 0
    Tsb = mr.FKinSpace(M_home, Slist, thetalist)
    Vs  = mr.Adjoint(Tsb) @ mr.se3ToVec(mr.MatrixLog6(mr.TransInv(Tsb) @ T_target))
    err = np.linalg.norm(Vs[:3]) > eomg or np.linalg.norm(Vs[3:]) > ev

    while err and i < max_iter:
        thetalist += np.linalg.pinv(mr.JacobianSpace(Slist, thetalist)) @ Vs
        i   += 1
        Tsb  = mr.FKinSpace(M_home, Slist, thetalist)
        Vs   = mr.Adjoint(Tsb) @ mr.se3ToVec(mr.MatrixLog6(mr.TransInv(Tsb) @ T_target))
        err  = np.linalg.norm(Vs[:3]) > eomg or np.linalg.norm(Vs[3:]) > ev

    return thetalist, not err, i
