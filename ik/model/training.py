"""
Training and evaluation routines for the IK MLP.

Loss:
  Primary:  MSE of predicted q (normalised) vs ground-truth y — backpropagated.
  Display:  Position error  ||FK(q_pred)[:3,3] - P_target||
            Rotation error  ||R_pred_6          - R6_target||  (first 2 rows, 6-D)
  Display losses are computed with no_grad and are NOT backpropagated.
"""

import torch
import torch.nn as nn
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


def _display_losses(q_pred_raw, P_target, R6_target):
    """FK on predicted q only; compare against raw targets directly."""
    T_pred   = FK_batch_full(q_pred_raw)
    pos_disp = torch.mean(torch.norm(T_pred[:, :3, 3] - P_target, dim=1))
    R_pred_6 = T_pred[:, :2, :3].reshape(len(q_pred_raw), 6)
    rot_disp = torch.mean(torch.norm(R_pred_6 - R6_target, dim=1))
    return pos_disp, rot_disp


# ---------------------------------------------------------------------------
# Core train / eval loops
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device,
    MinMax_Y: list,
) -> tuple:
    """Run one training epoch. Returns (mse_loss, pos_display, rot_display)."""
    model.train()
    Y_min, Y_max = _to_tensors(MinMax_Y, device)
    criterion = nn.MSELoss()
    total_mse, pos_sum, rot_sum = 0.0, 0.0, 0.0

    for X, y, _, _, P_target, R6_target in loader:
        X, y      = X.to(device), y.to(device)
        P_target  = P_target.to(device)
        R6_target = R6_target.to(device)

        optimizer.zero_grad()
        y_pred = model(X)
        loss   = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            q_pred_raw         = _denorm_q(y_pred.detach(), Y_min, Y_max)
            pos_disp, rot_disp = _display_losses(q_pred_raw, P_target, R6_target)

        n = len(X)
        total_mse += loss.item() * n
        pos_sum   += pos_disp.item() * n
        rot_sum   += rot_disp.item() * n

    N = len(loader.dataset)
    return total_mse / N, pos_sum / N, rot_sum / N


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device,
    MinMax_Y: list,
) -> tuple:
    """Evaluate on a loader without gradients. Returns (mse_loss, pos_display, rot_display)."""
    model.eval()
    Y_min, Y_max = _to_tensors(MinMax_Y, device)
    criterion = nn.MSELoss()
    total_mse, pos_sum, rot_sum = 0.0, 0.0, 0.0

    with torch.no_grad():
        for X, y, _, _, P_target, R6_target in loader:
            X, y      = X.to(device), y.to(device)
            P_target  = P_target.to(device)
            R6_target = R6_target.to(device)

            y_pred             = model(X)
            loss               = criterion(y_pred, y)
            q_pred_raw         = _denorm_q(y_pred, Y_min, Y_max)
            pos_disp, rot_disp = _display_losses(q_pred_raw, P_target, R6_target)

            n = len(X)
            total_mse += loss.item() * n
            pos_sum   += pos_disp.item() * n
            rot_sum   += rot_disp.item() * n

    N = len(loader.dataset)
    return total_mse / N, pos_sum / N, rot_sum / N


def evaluate_and_return_loss(
    model: nn.Module,
    loader: DataLoader,
    device,
    MinMax_Y: list,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-sample (mse_losses, pos_losses, rot_losses) as 1-D tensors. Prints means."""
    model.eval()
    Y_min, Y_max = _to_tensors(MinMax_Y, device)
    all_mse, all_pos, all_rot = [], [], []

    with torch.no_grad():
        for X, y, _, _, P_target, R6_target in loader:
            X, y      = X.to(device), y.to(device)
            P_target  = P_target.to(device)
            R6_target = R6_target.to(device)

            y_pred     = model(X)
            mse        = (y_pred - y).pow(2).mean(dim=1)
            q_pred_raw = _denorm_q(y_pred, Y_min, Y_max)

            T_pred   = FK_batch_full(q_pred_raw)
            pos_loss = torch.norm(T_pred[:, :3, 3] - P_target, dim=1)
            R_pred_6 = T_pred[:, :2, :3].reshape(len(X), 6)
            rot_loss = torch.norm(R_pred_6 - R6_target, dim=1)

            all_mse.append(mse)
            all_pos.append(pos_loss)
            all_rot.append(rot_loss)

    mse_losses = torch.cat(all_mse)
    pos_losses = torch.cat(all_pos)
    rot_losses = torch.cat(all_rot)

    print(
        f"mean mse loss: {mse_losses.mean().item():.4f} | "
        f"mean pos loss: {pos_losses.mean().item():.4f} | "
        f"mean rot loss: {rot_losses.mean().item():.4f}"
    )
    return mse_losses, pos_losses, rot_losses


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
) -> nn.Module:
    """
    Full training loop with ReduceLROnPlateau scheduler and early stopping.
    Primary loss: MSE on normalised q.
    Returns model with the best weights loaded.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val       = float("inf")
    no_improvement = 0

    for epoch in range(1, epochs + 1):
        train_mse, train_pos, train_rot = train(model, train_loader, optimizer, device, MinMax_Y)
        val_mse,   val_pos,   val_rot   = evaluate(model, val_loader, device, MinMax_Y)
        scheduler.step(val_mse)

        if val_mse < best_val:
            best_val       = val_mse
            torch.save(model.state_dict(), save_path)
            no_improvement = 0
        else:
            no_improvement += 1

        print(
            f"epoch {epoch:3d} | "
            f"train mse {train_mse:.4f} (pos {train_pos:.4f} rot {train_rot:.4f}) | "
            f"val mse {val_mse:.4f} (pos {val_pos:.4f} rot {val_rot:.4f}) | "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"\nbest val mse: {best_val:.4f}")
    print(f"model saved to {save_path}")
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
