"""
IK inference — MLP warm start + Newton-Raphson refinement.

Usage
-----
Single query:
    from ik.inference import IKSolver
    solver = IKSolver()
    result = solver.solve(q_current, T_target)
    print(result["q"], result["converged"], result["iterations"])

Batch:
    results = solver.solve_batch(q_currents, T_targets)
    for r in results:
        print(r["q"], r["converged"], r["iterations"])

Inputs
------
q_current  : array-like (6,)    — current joint angles in radians
T_target   : array-like (4, 4)  — desired end-effector homogeneous transform
q_currents : array-like (N, 6)
T_targets  : array-like (N, 4, 4)
"""

import json
import numpy as np
import torch
from pathlib import Path

import warnings
import modern_robotics as mr

from ik.model.mlp import MLP
from ik.viperx300 import SLIST, M_HOME, JOINT_LIMITS


# ---------------------------------------------------------------------------
# Repo-relative defaults (read from config.json)
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    return json.loads(config_path.read_text())


class IKSolver:
    """
    Load once, call many times.

    Parameters
    ----------
    model_path : path to the .pt weights file.
                 Defaults to config.json → SAVE_MODEL.
    minmax_dir : directory containing MinMax_X.npy and MinMax_Y.npy.
                 Defaults to the same folder as model_path.
    device     : torch device string. Auto-detects CUDA if None.
    """

    def __init__(self, model_path=None, minmax_dir=None, device=None):
        cfg = _load_config()
        repo_root = Path(__file__).resolve().parent.parent

        # Paths
        if model_path is None:
            model_path = repo_root / cfg["SAVE_MODEL"]
        else:
            model_path = Path(model_path)

        if minmax_dir is None:
            minmax_dir = model_path.parent

        # NR tolerances from config
        self._eomg    = cfg["NR_EOMG"]
        self._ev      = cfg["NR_EV"]
        self._maxiter = cfg["NR_MAXITER"]

        # Device
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Normalisation stats
        _mx = np.load(minmax_dir / "MinMax_X.npy", allow_pickle=True)
        _my = np.load(minmax_dir / "MinMax_Y.npy", allow_pickle=True)
        self._MinMax_X = [np.array(_mx[0], dtype=np.float32), np.array(_mx[1], dtype=np.float32)]
        self._MinMax_Y = [np.array(_my[0], dtype=np.float32), np.array(_my[1], dtype=np.float32)]

        # Model
        self._model = MLP()
        self._model.load_state_dict(torch.load(model_path, map_location=self._device))
        self._model.to(self._device).eval()

        print(f"IKSolver ready | device: {self._device} | model: {model_path.name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, q_current, T_target) -> dict:
        """
        Solve IK for a single configuration.

        Returns
        -------
        dict with keys:
            q          : (6,) ndarray — solution joint angles in radians
            converged  : bool
            iterations : int
            q_mlp      : (6,) ndarray — raw MLP prediction before NR
        """
        q_current = np.array(q_current, dtype=np.float32).flatten()
        T_target  = np.array(T_target,  dtype=np.float64)
        assert q_current.shape == (6,),   "q_current must be shape (6,)"
        assert T_target.shape  == (4, 4), "T_target must be shape (4, 4)"
        self._warn_limits(q_current)

        q_mlp = self._mlp_predict(q_current, T_target)
        q, converged, iterations = self._nr_ik(T_target, q_mlp)
        return {"q": q, "converged": converged, "iterations": iterations, "q_mlp": q_mlp}

    def solve_batch(self, q_currents, T_targets) -> list[dict]:
        """
        Solve IK for a batch of (q_current, T_target) pairs.

        Parameters
        ----------
        q_currents : (N, 6)
        T_targets  : (N, 4, 4)

        Returns
        -------
        List of N dicts, each with keys: q, converged, iterations, q_mlp
        """
        q_currents = np.array(q_currents, dtype=np.float32)
        T_targets  = np.array(T_targets,  dtype=np.float64)
        assert q_currents.ndim == 2 and q_currents.shape[1] == 6
        assert T_targets.ndim  == 3 and T_targets.shape[1:] == (4, 4)
        assert len(q_currents) == len(T_targets)
        for i, q in enumerate(q_currents):
            self._warn_limits(q, index=i)

        q_mlps = self._mlp_predict_batch(q_currents, T_targets)

        results = []
        for q_mlp, T in zip(q_mlps, T_targets):
            q, converged, iterations = self._nr_ik(T, q_mlp)
            results.append({"q": q, "converged": converged,
                            "iterations": iterations, "q_mlp": q_mlp})
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _warn_limits(self, q: np.ndarray, index: int = None) -> None:
        violations = np.where((q < JOINT_LIMITS[:, 0]) | (q > JOINT_LIMITS[:, 1]))[0]
        if len(violations) == 0:
            return
        prefix = f"q_currents[{index}]" if index is not None else "q_current"
        joint_names = ["waist", "shoulder", "elbow", "forearm roll", "wrist angle", "wrist rotate"]
        for j in violations:
            warnings.warn(
                f"{prefix} joint {j} ({joint_names[j]}) = {np.degrees(q[j]):.1f}° is outside "
                f"limits [{np.degrees(JOINT_LIMITS[j, 0]):.1f}°, {np.degrees(JOINT_LIMITS[j, 1]):.1f}°]",
                UserWarning, stacklevel=3,
            )

    def _build_x(self, q_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
        """Build and normalise a single 15-dim input vector."""
        R6 = T_target[:2, :3].flatten().astype(np.float32)
        P  = T_target[:3, 3].astype(np.float32)
        x  = np.concatenate([R6, P, q_current])
        x  = (x - self._MinMax_X[0]) / (self._MinMax_X[1] - self._MinMax_X[0])
        return (x * 2 - 1).astype(np.float32)

    def _denorm_y(self, y: np.ndarray) -> np.ndarray:
        return (y + 1) / 2 * (self._MinMax_Y[1] - self._MinMax_Y[0]) + self._MinMax_Y[0]

    def _mlp_predict(self, q_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
        x = self._build_x(q_current, T_target)
        with torch.no_grad():
            x_t = torch.tensor(x).unsqueeze(0).to(self._device)
            y   = self._model(x_t).squeeze(0).cpu().numpy()
        return self._denorm_y(y)

    def _mlp_predict_batch(self, q_currents: np.ndarray, T_targets: np.ndarray) -> np.ndarray:
        xs = np.stack([self._build_x(q, T) for q, T in zip(q_currents, T_targets)])
        with torch.no_grad():
            x_t = torch.tensor(xs).to(self._device)
            ys  = self._model(x_t).cpu().numpy()
        return self._denorm_y(ys)

    def _nr_ik(self, T_target: np.ndarray, q_init: np.ndarray):
        # taken from mdodern robotics package
        thetalist = np.array(q_init, dtype=float).copy()
        i   = 0
        Tsb = mr.FKinSpace(M_HOME, SLIST, thetalist)
        Vs  = mr.Adjoint(Tsb) @ mr.se3ToVec(mr.MatrixLog6(mr.TransInv(Tsb) @ T_target))
        err = np.linalg.norm(Vs[:3]) > self._eomg or np.linalg.norm(Vs[3:]) > self._ev

        while err and i < self._maxiter:
            thetalist += np.linalg.pinv(mr.JacobianSpace(SLIST, thetalist)) @ Vs
            i   += 1
            Tsb  = mr.FKinSpace(M_HOME, SLIST, thetalist)
            Vs   = mr.Adjoint(Tsb) @ mr.se3ToVec(mr.MatrixLog6(mr.TransInv(Tsb) @ T_target))
            err  = np.linalg.norm(Vs[:3]) > self._eomg or np.linalg.norm(Vs[3:]) > self._ev

        return thetalist, not err, i


# ---------------------------------------------------------------------------
# CLI — quick test from terminal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from ik.kinematics.fk import FK
    from ik.viperx300 import JOINT_LIMITS

    np.random.seed(0)
    solver = IKSolver()

    print("\n--- Single solve ---")
    q_true = np.random.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    T      = FK(q_true)
    q_init = np.random.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    r      = solver.solve(q_init, T)
    print(f"  converged : {r['converged']}")
    print(f"  iterations: {r['iterations']}")
    print(f"  q_solution: {np.round(r['q'], 4)}")

    print("\n--- Batch solve (N=5) ---")
    N         = 5
    q_trues   = np.random.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1], (N, 6))
    T_targets = np.array([FK(q) for q in q_trues])
    q_inits   = np.random.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1], (N, 6))
    results   = solver.solve_batch(q_inits, T_targets)
    for i, r in enumerate(results):
        print(f"  [{i}] converged={r['converged']}  iters={r['iterations']}")
