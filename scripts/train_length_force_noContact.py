import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import touchbot_controller
import time
# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH = "data/size3_data_noContact.pkl"
SAVE_PATH = "NN_models/size3_length_force_noContact.pt"

# ── Model ─────────────────────────────────────────────────────────────────────
class LengthForceNet(nn.Module):
    """
    3 hidden layers × 64 units, ~8,900 parameters.
    Despite 504 samples, smooth physics data benefits from higher capacity;
    overfitting is controlled via Adam weight decay instead of dropout.
    Input : delta cable lengths (4,)
    Output: cable forces        (4,)
    """
    def __init__(self, in_dim=4, out_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(path=DATA_PATH):
    """Returns normalised X (cable lengths) and raw y (forces), plus X stats for inference."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    cable_lengths = np.array(data["cable_lengths"])            # (N, 4)
    forces        = np.array(data["forces"])                   # (N, 4)

    print(f"Samples : {len(cable_lengths)}")
    print(f"Input   : cable_lengths  range=[{cable_lengths.min():.3f}, {cable_lengths.max():.3f}]")
    print(f"Output  : forces         range=[{forces.min():.3f}, {forces.max():.3f}]")

    X_mean, X_std = cable_lengths.mean(0), cable_lengths.std(0) + 1e-8
    X_norm = (cable_lengths - X_mean) / X_std

    X = torch.tensor(X_norm,  dtype=torch.float32)
    y = torch.tensor(forces,  dtype=torch.float32)
    stats = dict(X_mean=X_mean, X_std=X_std)
    return X, y, stats


# ── Training ──────────────────────────────────────────────────────────────────
def train(
    X, y,
    hidden=64,
    epochs=500,
    lr=1e-3,
    batch_size=32,
    val_split=0.2,
    weight_decay=1e-4,
    seed=42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # split
    dataset = TensorDataset(X, y)
    n_val   = int(val_split * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=len(val_ds))

    # model
    model     = LengthForceNet(hidden=hidden).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params}  (train samples: {n_train}  →  ratio {n_train/n_params:.1f}×)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None

    print(f"Training for {epochs} epochs …")
    for epoch in range(1, epochs + 1):
        # train step
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        # validation step
        model.eval()
        with torch.no_grad():
            xv, yv   = next(iter(val_loader))
            xv, yv   = xv.to(device), yv.to(device)
            val_loss = criterion(model(xv), yv).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

    model.load_state_dict(best_state)
    print(f"\nBest val loss: {best_val_loss:.6f}")
    return model, best_val_loss


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, X, y_true, device=None):
    """Print MAE / RMSE in force units."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pred = model(X.to(device)).cpu().numpy()

    truth = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true

    mae  = np.abs(pred - truth).mean()
    rmse = np.sqrt(((pred - truth) ** 2).mean())
    mse  = ((pred - truth) ** 2).mean()
    print(f"\nFull-dataset evaluation:")
    print(f"  MSE  = {mse:.5f}")
    print(f"  MAE  = {mae:.5f}")
    print(f"  RMSE = {rmse:.5f}")
    print(f"  Per-cable MAE:")
    for i in range(truth.shape[1]):
        print(f"    Cable {i}: {np.abs(pred[:, i] - truth[:, i]).mean():.5f}")


# ── Save ──────────────────────────────────────────────────────────────────────
def save_checkpoint(model, stats, path=SAVE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model_state": model.state_dict()}, path)
    stats_path = path.replace(".pt", "_stats.npz")
    np.savez(stats_path, **stats)
    print(f"Checkpoint saved → {path}")
    print(f"Input stats saved → {stats_path}")


# ── Inference ─────────────────────────────────────────────────────────────────
def load_checkpoint(path=SAVE_PATH):
    """Load model and input normalisation stats from a saved checkpoint."""
    ckpt  = torch.load(path, map_location="cpu", weights_only=True)
    model = LengthForceNet()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    stats_path = path.replace(".pt", "_stats.npz")
    raw   = np.load(stats_path)
    stats = {k: raw[k] for k in ("X_mean", "X_std")}
    return model, stats


def infer(cable_lengths, model, stats, device=None):
    """
    Predict cable forces from absolute cable lengths.

    Parameters
    ----------
    cable_lengths : array-like, shape (4,) or (N, 4)
        absolute cable lengths (not delta)

    Returns
    -------
    forces : np.ndarray, same leading shape as input, units = N
    """
    if device is None:
        device = next(model.parameters()).device

    x = np.asarray(cable_lengths, dtype=np.float32)
    scalar = x.ndim == 1
    if scalar:
        x = x[None]                                       # (1, 4)

    x_norm = ((x - stats["X_mean"]) / stats["X_std"]).astype(np.float32)
    with torch.no_grad():
        forces = model(torch.tensor(x_norm).to(device)).cpu().numpy()

    forces = np.clip(forces, 0.0, None)   # forces are physically non-negative
    return forces[0] if scalar else forces


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    X, y, stats = load_data()

    model, _ = train(
        X, y,
        hidden=64,
        epochs=500,
        lr=1e-3,
        batch_size=32,
        weight_decay=1e-4,
    )

    evaluate(model, X, y)
    save_checkpoint(model, stats)




if __name__ == "__main__":
    touchbot = touchbot_controller.touchbotController()
    time.sleep(0.1)
    touchbot.calibrate_motor_pos()
    touchbot.checkpoint("Calibration complete, start collecting data")
    # touchbot.send_length_cmd_rel_timed([-23.45, -43.51, -36.81, -18.13],5)
    touchbot.send_length_cmd_abs_timed([92, 104, 108, 99], 5)
    # touchbot.send_length_cmd_rel_timed([-1.8, -3.1, -3.1, -2.5])
    time.sleep(0.1)
    # touchbot.enforce_min_cable_forces([0.1,0.1,0.1,0.1])
    touchbot.checkpoint("now compare real world data and NN model")
    cur_cl = touchbot.get_state()["cable_lengths"]
    force_reading = touchbot.cts.read_forces()
    print("Real forces:")
    print(force_reading)
    print("cur_cl:")
    print(cur_cl)
    model, stats = load_checkpoint()
    forces_inferred = forces = infer(cur_cl, model, stats)
    print("Inferred forces:")
    print(forces_inferred)
    print("differences")
    print(np.array(force_reading) - np.array(forces_inferred))
    touchbot.close_all()
