"""
Training and evaluation routines for the IK MLP.

Loss:
  alpha_pos * ||FK(q_pred)[:3,3] - P_target||   — task-space position
  alpha_rot * ||R_pred_6 - R6_target||           — task-space rotation
"""

import torch
import torch.nn as nn  # kept for type hints
from torch.utils.data import DataLoader

from ik.kinematics.fk import FK_batch_full


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensors(MinMax_Y: list, device):
    Y_min = torch.as_tensor(MinMax_Y[0], dtype=torch.float32).to(device)
    Y_max = torch.as_tensor(MinMax_Y[1], dtype=torch.float32).to(device)
    return Y_min, Y_max


def _denorm_q(y_pred: torch.Tensor, Y_min: torch.Tensor, Y_max: torch.Tensor) -> torch.Tensor:
    """Inverse of MinMax [-1, 1] normalisation."""
    return (y_pred + 1) / 2 * (Y_max - Y_min) + Y_min


def _task_space_losses(q_pred_raw, P_target, R6_target):
    """
    Compute batched position and rotation errors from raw (denormalised) q_pred.
    Differentiable — call inside autograd context for backprop, or with no_grad for display.
    """
    T_pred   = FK_batch_full(q_pred_raw)
    pos_loss = torch.mean(torch.norm(T_pred[:, :3, 3] - P_target, dim=1))
    R_pred_6 = T_pred[:, :2, :3].reshape(len(q_pred_raw), 6)
    rot_loss = torch.mean(torch.norm(R_pred_6 - R6_target, dim=1))
    return pos_loss, rot_loss


# ---------------------------------------------------------------------------
# Core train / eval loops
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device,
    MinMax_Y: list,
    alpha_pos: float = 1.0,
    alpha_rot: float = 0.1,
) -> tuple:
    """
    Run one training epoch.
    Loss = MSE(q_norm) + alpha_pos * pos_err + alpha_rot * rot_err.
    Returns (total_loss, pos_loss, rot_loss) epoch means.
    """
    model.train()
    Y_min, Y_max = _to_tensors(MinMax_Y, device)
    total_loss, pos_sum, rot_sum = 0.0, 0.0, 0.0

    for X, y, _, _, P_target, R6_target in loader:
        X         = X.to(device)
        P_target  = P_target.to(device)
        R6_target = R6_target.to(device)

        optimizer.zero_grad()
        q_pred_raw         = _denorm_q(model(X), Y_min, Y_max)
        pos_loss, rot_loss = _task_space_losses(q_pred_raw, P_target, R6_target)
        loss = alpha_pos * pos_loss + alpha_rot * rot_loss
        loss.backward()
        optimizer.step()

        n = len(X)
        total_loss += loss.item() * n
        pos_sum    += pos_loss.item() * n
        rot_sum    += rot_loss.item() * n

    N = len(loader.dataset)
    return total_loss / N, pos_sum / N, rot_sum / N


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device,
    MinMax_Y: list,
    alpha_pos: float = 1.0,
    alpha_rot: float = 0.1,
) -> tuple:
    """Evaluate on a loader without gradients. Returns (total_loss, pos_loss, rot_loss)."""
    model.eval()
    Y_min, Y_max = _to_tensors(MinMax_Y, device)
    total_loss, pos_sum, rot_sum = 0.0, 0.0, 0.0

    with torch.no_grad():
        for X, y, _, _, P_target, R6_target in loader:
            X         = X.to(device)
            P_target  = P_target.to(device)
            R6_target = R6_target.to(device)

            q_pred_raw         = _denorm_q(model(X), Y_min, Y_max)
            pos_loss, rot_loss = _task_space_losses(q_pred_raw, P_target, R6_target)
            loss               = alpha_pos * pos_loss + alpha_rot * rot_loss

            n = len(X)
            total_loss += loss.item() * n
            pos_sum    += pos_loss.item() * n
            rot_sum    += rot_loss.item() * n

    N = len(loader.dataset)
    return total_loss / N, pos_sum / N, rot_sum / N


def evaluate_and_return_loss(
    model: nn.Module,
    loader: DataLoader,
    device,
    MinMax_Y: list,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample (pos_losses, rot_losses) as 1-D tensors. Prints means."""
    model.eval()
    Y_min, Y_max = _to_tensors(MinMax_Y, device)
    all_pos, all_rot = [], []

    with torch.no_grad():
        for X, y, _, _, P_target, R6_target in loader:
            X         = X.to(device)
            P_target  = P_target.to(device)
            R6_target = R6_target.to(device)

            q_pred_raw = _denorm_q(model(X), Y_min, Y_max)
            T_pred     = FK_batch_full(q_pred_raw)
            pos_loss   = torch.norm(T_pred[:, :3, 3] - P_target, dim=1)
            R_pred_6   = T_pred[:, :2, :3].reshape(len(X), 6)
            rot_loss   = torch.norm(R_pred_6 - R6_target, dim=1)

            all_pos.append(pos_loss)
            all_rot.append(rot_loss)

    pos_losses = torch.cat(all_pos)
    rot_losses = torch.cat(all_rot)

    print(
        f"mean pos loss: {pos_losses.mean().item():.4f} m | "
        f"mean rot loss: {rot_losses.mean().item():.4f}"
    )
    return pos_losses, rot_losses


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    MinMax_Y: list,
    device,
    save_path: str,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 30,
    alpha_pos: float = 1.0,
    alpha_rot: float = 0.1,
    MinMax_X: list = None,  # required — kept as kwarg for call-site clarity
) -> nn.Module:
    """
    Full training loop with ReduceLROnPlateau scheduler and early stopping.
    Loss = MSE(q_norm) + alpha_pos * pos_err + alpha_rot * rot_err.
    Returns model with the best weights loaded.

    Saves MinMax_X.npy and MinMax_Y.npy next to save_path so inference
    scripts can load normalisation stats without touching the dataset.
    """
    if MinMax_X is None:
        raise ValueError("MinMax_X is required — pass train_ds.MinMax_X")
    import os, numpy as np

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val       = float("inf")
    no_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_pos, train_rot = train(
            model, train_loader, optimizer, device, MinMax_Y, alpha_pos, alpha_rot
        )
        val_loss, val_pos, val_rot = evaluate(
            model, val_loader, device, MinMax_Y, alpha_pos, alpha_rot
        )
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val       = val_loss
            torch.save(model.state_dict(), save_path)
            no_improvement = 0
        else:
            no_improvement += 1

        print(
            f"epoch {epoch:3d} | "
            f"train {train_loss:.4f} (pos {train_pos:.4f} rot {train_rot:.4f}) | "
            f"val {val_loss:.4f} (pos {val_pos:.4f} rot {val_rot:.4f}) | "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"\nbest val loss: {best_val:.4f}")
    print(f"model saved to {save_path}")

    model_dir = os.path.dirname(save_path)
    np.save(os.path.join(model_dir, "MinMax_X.npy"), np.array(MinMax_X, dtype=object))
    np.save(os.path.join(model_dir, "MinMax_Y.npy"), np.array(MinMax_Y, dtype=object))
    print(f"MinMax saved to {model_dir}")

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from ik.data.dataset import IKDataset
    from ik.model.mlp import MLP

    SAVE_DIR = "G:/My Drive/inverse_kinematics/hot_start"
    DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {DEVICE}")

    train_ds = IKDataset("train", SAVE_DIR)
    val_ds   = IKDataset("val",   SAVE_DIR,
                         MinMax_X=train_ds.MinMax_X, MinMax_Y=train_ds.MinMax_Y)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=2)

    model = MLP(input_dim=15).to(DEVICE)  # 6 (R6) + 3 (P) + 6 (q_init) = 15
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"train samples:    {len(train_ds):,}")
    print(f"val samples:      {len(val_ds):,}\n")

    run_training(
        model, train_loader, val_loader,
        MinMax_Y=train_ds.MinMax_Y,
        device=DEVICE,
        save_path=f"{SAVE_DIR}/mlp_ik.pt",
    )
