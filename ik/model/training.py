"""
Training and evaluation routines for the IK MLP.

Two modes controlled by hot_start flag:

  hot_start=False (local-Jacobian):
    q_pred = q1 + dq_pred
    loss   = ||FK(q_pred)[:3,3] - xd||

  hot_start=True:
    q_pred = model output (absolute q)
    loss   = ||p_pred - p_target|| + lambda_rot * ||R_pred - R_target||_F
    where R_target comes from FK(q1) — q1 IS the target config in this mode.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ik.kinematics.fk import FK_batch, FK_batch_full


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def task_space_loss(
    q_pred: torch.Tensor,
    q_target: torch.Tensor,
    xd: torch.Tensor,
    hot_start: bool = False,
    lambda_rot: float = 1.0,
) -> torch.Tensor:
    """
    Args:
        q_pred:    (B, 6) predicted joint config
        q_target:  (B, 6) target joint config (q1 in the dataset)
        xd:        (B, 3) target EE position
        hot_start: if True adds Frobenius rotation loss
        lambda_rot: weight for rotation term (only used when hot_start=True)
    """
    if hot_start:
        T_pred   = FK_batch_full(q_pred)
        T_target = FK_batch_full(q_target)
        pos_loss = torch.mean(torch.norm(T_pred[:, :3, 3] - T_target[:, :3, 3], dim=1))
        rot_loss = torch.mean(torch.norm(T_pred[:, :3, :3] - T_target[:, :3, :3], dim=(1, 2)))
        return pos_loss + lambda_rot * rot_loss, pos_loss, rot_loss
    else:
        pos_loss = torch.mean(torch.norm(FK_batch(q_pred) - xd, dim=1))
        return pos_loss, pos_loss, torch.zeros(1, device=q_pred.device)


# ---------------------------------------------------------------------------
# Core train / eval loops
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device,
    scaler_Y: list,
    hot_start: bool = False,
    lambda_rot: float = 1.0,
) -> float:
    """Run one training epoch. Returns (total, pos, rot) mean losses."""
    model.train()
    Y_mean, Y_std = scaler_Y[0].to(device), scaler_Y[1].to(device)
    total_loss, pos_loss_sum, rot_loss_sum = 0.0, 0.0, 0.0

    for X, _, q1, xd in loader:
        X, q1, xd = X.to(device), q1.to(device), xd.to(device)
        optimizer.zero_grad()

        q_pred = model(X) * Y_std + Y_mean
        if not hot_start:
            q_pred = q1 + q_pred   # local-Jacobian: output is delta

        loss, pos, rot = task_space_loss(q_pred, q1, xd, hot_start, lambda_rot)
        loss.backward()
        optimizer.step()
        n = len(X)
        total_loss   += loss.item() * n
        pos_loss_sum += pos.item()  * n
        rot_loss_sum += rot.item()  * n

    N = len(loader.dataset)
    return total_loss / N, pos_loss_sum / N, rot_loss_sum / N


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device,
    scaler_Y: list,
    hot_start: bool = False,
    lambda_rot: float = 1.0,
) -> float:
    """Evaluate on a loader without gradients. Returns (total, pos, rot) mean losses."""
    model.eval()
    Y_mean, Y_std = scaler_Y[0].to(device), scaler_Y[1].to(device)
    total_loss, pos_loss_sum, rot_loss_sum = 0.0, 0.0, 0.0

    with torch.no_grad():
        for X, _, q1, xd in loader:
            X, q1, xd = X.to(device), q1.to(device), xd.to(device)

            q_pred = model(X) * Y_std + Y_mean
            if not hot_start:
                q_pred = q1 + q_pred

            loss, pos, rot = task_space_loss(q_pred, q1, xd, hot_start, lambda_rot)
            n = len(X)
            total_loss   += loss.item() * n
            pos_loss_sum += pos.item()  * n
            rot_loss_sum += rot.item()  * n

    N = len(loader.dataset)
    return total_loss / N, pos_loss_sum / N, rot_loss_sum / N


def evaluate_and_return_loss(
    model: nn.Module,
    loader: DataLoader,
    device,
    scaler_Y: list,
    hot_start: bool = False,
    lambda_rot: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-sample (pos_losses, rot_losses, total_losses) as 1-D tensors.
    rot_losses is zeros when hot_start=False. Prints mean of all three."""
    model.eval()
    Y_mean, Y_std = scaler_Y[0].to(device), scaler_Y[1].to(device)
    all_pos, all_rot, all_total = [], [], []

    with torch.no_grad():
        for X, _, q1, xd in loader:
            X, q1, xd = X.to(device), q1.to(device), xd.to(device)

            q_pred = model(X) * Y_std + Y_mean
            if not hot_start:
                q_pred = q1 + q_pred

            if hot_start:
                T_pred   = FK_batch_full(q_pred)
                T_target = FK_batch_full(q1)
                pos_loss = torch.norm(T_pred[:, :3, 3] - T_target[:, :3, 3], dim=1)
                rot_loss = torch.norm(T_pred[:, :3, :3] - T_target[:, :3, :3], dim=(1, 2))
                total    = pos_loss + lambda_rot * rot_loss
            else:
                pos_loss = torch.norm(FK_batch(q_pred) - xd, dim=1)
                rot_loss = torch.zeros_like(pos_loss)
                total    = pos_loss

            all_pos.append(pos_loss)
            all_rot.append(rot_loss)
            all_total.append(total)

    pos_losses   = torch.cat(all_pos)
    rot_losses   = torch.cat(all_rot)
    total_losses = torch.cat(all_total)

    print(
        f"mean pos loss:   {pos_losses.mean().item():.4f} | "
        f"mean rot loss:   {rot_losses.mean().item():.4f} | "
        f"mean total loss: {total_losses.mean().item():.4f}"
    )

    return pos_losses, rot_losses, total_losses


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scaler_Y: list,
    device,
    save_path: str,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 30,
    hot_start: bool = False,
    lambda_rot: float = 1.0,
) -> nn.Module:
    """
    Full training loop with ReduceLROnPlateau scheduler and early stopping.

    Returns:
        model with the best weights loaded.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val       = float("inf")
    no_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_pos, train_rot = train(model, train_loader, optimizer, device, scaler_Y, hot_start, lambda_rot)
        val_loss,   val_pos,   val_rot   = evaluate(model, val_loader, device, scaler_Y, hot_start, lambda_rot)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
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

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from ik.data.dataset import IKDataset
    from ik.model.mlp import MLP

    HOT_START = True
    SAVE_DIR  = "G:/My Drive/inverse_kinematics/hot_start"
    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {DEVICE}")

    train_ds = IKDataset("train", SAVE_DIR, hot_start=HOT_START)
    val_ds   = IKDataset("val",   SAVE_DIR, hot_start=HOT_START,
                         scaler_X=train_ds.scaler_X, scaler_Y=train_ds.scaler_Y)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=2)

    input_dim = 3 if HOT_START else 9
    model = MLP(input_dim=input_dim).to(DEVICE)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"train samples:    {len(train_ds):,}")
    print(f"val samples:      {len(val_ds):,}\n")

    run_training(
        model, train_loader, val_loader,
        scaler_Y=train_ds.scaler_Y,
        device=DEVICE,
        save_path=f"{SAVE_DIR}/mlp_hot_start.pt",
        hot_start=HOT_START,
        lambda_rot=1.0,
    )
