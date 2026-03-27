"""
Training and evaluation routines for the IK MLP.

Loss: mean end-effector position error  ||FK(q1 + dq_pred) - x_d||
using the differentiable FK_batch so gradients flow through the FK.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ik.kinematics.fk import FK_batch


# ---------------------------------------------------------------------------
# Core train / eval loops
# ---------------------------------------------------------------------------

def train(model: nn.Module, loader: DataLoader, optimizer, device, scaler_Y: list) -> float:
    """
    Run one training epoch.

    Returns:
        Mean task-space position error in metres over the epoch.
    """
    model.train()
    Y_mean, Y_std = scaler_Y[0].to(device), scaler_Y[1].to(device)
    total_loss = 0.0

    for X, _, q1, xd in loader:
        X, q1, xd = X.to(device), q1.to(device), xd.to(device)
        optimizer.zero_grad()

        dq_pred  = model(X) * Y_std + Y_mean
        x_reached = FK_batch(q1 + dq_pred)
        loss = torch.mean(torch.norm(x_reached - xd, dim=1))

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)

    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device, scaler_Y: list) -> float:
    """
    Evaluate the model on a loader without computing gradients.

    Returns:
        Mean task-space position error in metres.
    """
    model.eval()
    Y_mean, Y_std = scaler_Y[0].to(device), scaler_Y[1].to(device)
    total_loss = 0.0

    with torch.no_grad():
        for X, _, q1, xd in loader:
            X, q1, xd = X.to(device), q1.to(device), xd.to(device)
            dq_pred   = model(X) * Y_std + Y_mean
            x_reached = FK_batch(q1 + dq_pred)
            loss = torch.mean(torch.norm(x_reached - xd, dim=1))
            total_loss += loss.item() * len(X)

    return total_loss / len(loader.dataset)


def evaluate_and_return_loss(
    model: nn.Module,
    loader: DataLoader,
    device,
    scaler_Y: list,
) -> torch.Tensor:
    """
    Evaluate the model and return per-sample position errors as a 1-D tensor.
    Useful for error analysis and plotting.
    """
    model.eval()
    Y_mean, Y_std = scaler_Y[0].to(device), scaler_Y[1].to(device)
    all_losses = []

    with torch.no_grad():
        for X, _, q1, xd in loader:
            X, q1, xd = X.to(device), q1.to(device), xd.to(device)
            dq_pred   = model(X) * Y_std + Y_mean
            x_reached = FK_batch(q1 + dq_pred)
            all_losses.append(torch.norm(x_reached - xd, dim=1))

    return torch.cat(all_losses)


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
) -> nn.Module:
    """
    Full training loop with ReduceLROnPlateau scheduler and early stopping.

    Args:
        model:        the MLP to train (already on device)
        train_loader: DataLoader for training split
        val_loader:   DataLoader for validation split
        scaler_Y:     [Y_mean, Y_std] tensors
        device:       torch device
        save_path:    where to save the best model checkpoint
        epochs:       maximum number of epochs
        lr:           initial learning rate for Adam
        patience:     early-stopping patience (epochs without val improvement)

    Returns:
        model with the best weights loaded.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val      = float("inf")
    no_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, device, scaler_Y)
        val_loss   = evaluate(model, val_loader, device, scaler_Y)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            no_improvement = 0
        else:
            no_improvement += 1

        print(
            f"epoch {epoch:3d} | "
            f"train: {train_loss * 1000:.3f} mm | "
            f"val: {val_loss * 1000:.3f} mm | "
            f"lr: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"\nbest val loss: {best_val * 1000:.3f} mm")
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

    SAVE_DIR = "/content/drive/MyDrive/inverse_kinematics"
    DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {DEVICE}")

    train_ds = IKDataset("train", SAVE_DIR)
    val_ds   = IKDataset("val",   SAVE_DIR, train_ds.scaler_X, train_ds.scaler_Y)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=2)

    model = MLP().to(DEVICE)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"train samples:    {len(train_ds):,}")
    print(f"val samples:      {len(val_ds):,}\n")

    run_training(
        model, train_loader, val_loader,
        scaler_Y=train_ds.scaler_Y,
        device=DEVICE,
        save_path=f"{SAVE_DIR}/mlp_best.pt",
    )
