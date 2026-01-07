"""
UE Localization with ML from ISAC-like channel parameters (AoA/AoD/ToA/Doppler/RSS).
- Generates synthetic dataset (replace with real measurements when available)
- Trains an MLP regressor to estimate UE (x, y)
- Saves model + scaler
"""

from __future__ import annotations

import math
import json
import numpy as np

from dataclasses import dataclass
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import joblib

# Use PyTorch for the ML model
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# -----------------------------
# 1) Synthetic ISAC-like dataset
# -----------------------------
@dataclass
class Scenario:
    # Base station (BS) position
    bs_xy: Tuple[float, float] = (0.0, 0.0)

    # Area where UE can be located
    ue_x_range: Tuple[float, float] = (-50.0, 50.0)
    ue_y_range: Tuple[float, float] = (5.0, 120.0)

    # Carrier & "measurement" noise levels (roughly representative)
    c: float = 3e8
    fc_hz: float = 28e9

    # Std dev of measurement noise
    aoa_noise_deg: float = 1.0       # degrees
    toa_noise_s: float = 2e-9        # seconds (2 ns ~ 0.6 m one-way)
    doppler_noise_hz: float = 20.0   # Hz
    rss_noise_db: float = 2.0        # dB

    # Motion: UE speed distribution (m/s)
    v_mean: float = 0.0
    v_std: float = 3.0


def _wrap_angle_deg(a: np.ndarray) -> np.ndarray:
    """Wrap angles to [-180, 180)."""
    out = (a + 180.0) % 360.0 - 180.0
    return out


def make_synthetic_dataset(
    n_samples: int = 50000,
    n_paths: int = 3,
    seed: int = 7,
    scenario: Scenario = Scenario(),
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Create synthetic features X and labels y=(x,y).
    Features per sample:
      - AoA for each path (deg)
      - ToA for each path (s)
      - Doppler for each path (Hz)
      - RSS for each path (dB)
    Shape: X = [n_samples, n_paths * 4], y = [n_samples, 2]
    """
    rng = np.random.default_rng(seed)
    bsx, bsy = scenario.bs_xy

    # UE positions
    ue_x = rng.uniform(*scenario.ue_x_range, size=n_samples)
    ue_y = rng.uniform(*scenario.ue_y_range, size=n_samples)
    y = np.stack([ue_x, ue_y], axis=1)

    # Basic LOS geometry
    dx = ue_x - bsx
    dy = ue_y - bsy
    dist = np.sqrt(dx**2 + dy**2) + 1e-9

    # AoA (BS receiving from UE) in degrees
    aoa_los = np.degrees(np.arctan2(dy, dx))  # [-180,180]
    aoa_los = _wrap_angle_deg(aoa_los)

    # ToA (one-way) in seconds
    toa_los = dist / scenario.c

    # Doppler (one-way) approx: fD = (v_rad / lambda)
    lam = scenario.c / scenario.fc_hz
    v = rng.normal(scenario.v_mean, scenario.v_std, size=n_samples)
    # radial component: assume motion roughly along line-of-sight (simple model)
    v_rad = v * rng.uniform(-1.0, 1.0, size=n_samples)
    doppler_los = v_rad / lam

    # RSS: free-space like path loss + noise (very simplified)
    # RSS(dB) ~ -20log10(dist) + const + noise
    rss_los = -20.0 * np.log10(dist) + rng.normal(0.0, scenario.rss_noise_db, size=n_samples)

    # Create multipath "extra paths" by perturbing LOS
    aoa = [aoa_los]
    toa = [toa_los]
    dop = [doppler_los]
    rss = [rss_los]

    for _ in range(1, n_paths):
        # multipath adds bias
        aoa.append(_wrap_angle_deg(aoa_los + rng.normal(0.0, 8.0, size=n_samples)))
        toa.append(toa_los + np.abs(rng.normal(0.0, 25e-9, size=n_samples)))  # extra delay
        dop.append(doppler_los + rng.normal(0.0, 80.0, size=n_samples))
        rss.append(rss_los - np.abs(rng.normal(0.0, 6.0, size=n_samples)))

    aoa = np.stack(aoa, axis=1) + rng.normal(0.0, scenario.aoa_noise_deg, size=(n_samples, n_paths))
    toa = np.stack(toa, axis=1) + rng.normal(0.0, scenario.toa_noise_s, size=(n_samples, n_paths))
    dop = np.stack(dop, axis=1) + rng.normal(0.0, scenario.doppler_noise_hz, size=(n_samples, n_paths))
    rss = np.stack(rss, axis=1) + rng.normal(0.0, scenario.rss_noise_db, size=(n_samples, n_paths))

    # Feature vector: [aoa1..aoaK, toa1..toaK, dop1..dopK, rss1..rssK]
    X = np.concatenate([aoa, toa, dop, rss], axis=1)

    meta = {
        "n_paths": n_paths,
        "features_order": ["aoa_deg", "toa_s", "doppler_hz", "rss_db"],
        "scenario": scenario.__dict__,
    }
    return X.astype(np.float32), y.astype(np.float32), meta


# -----------------------------
# 2) Model (MLP regressor)
# -----------------------------
class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [
                nn.Linear(d, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            d = hidden
        layers.append(nn.Linear(d, 2))  # output (x,y)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# 3) Train / Eval
# -----------------------------
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    out_prefix: str = "ue_loc_mlp",
    batch_size: int = 512,
    epochs: int = 25,
    lr: float = 1e-3,
    seed: int = 7,
) -> None:
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed)

    # Scale features (critical for MLP)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    train_ds = TensorDataset(torch.from_numpy(X_train_s).float(), torch.from_numpy(y_train).float())
    test_ds = TensorDataset(torch.from_numpy(X_test_s).float(), torch.from_numpy(y_test).float())

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = MLPRegressor(in_dim=X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()  # robust regression

    best_rmse = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_ds)

        # Eval
        model.eval()
        preds = []
        gts = []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy()
                preds.append(pred)
                gts.append(yb.numpy())
        preds = np.concatenate(preds, axis=0)
        gts = np.concatenate(gts, axis=0)

        rmse = math.sqrt(mean_squared_error(gts, preds))
        mae = mean_absolute_error(gts, preds)

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), f"{out_prefix}.pt")
            joblib.dump(scaler, f"{out_prefix}_scaler.joblib")

        print(f"[Epoch {ep:02d}] train_loss={tr_loss:.4f}  test_RMSE(m)={rmse:.3f}  test_MAE(m)={mae:.3f}")

    print(f"\nBest RMSE saved: {best_rmse:.3f} m")
    print(f"Saved model: {out_prefix}.pt")
    print(f"Saved scaler: {out_prefix}_scaler.joblib")


# -----------------------------
# 4) Inference helper
# -----------------------------
def load_and_predict(
    features: np.ndarray,
    model_path: str = "ue_loc_mlp.pt",
    scaler_path: str = "ue_loc_mlp_scaler.joblib",
) -> np.ndarray:
    """
    features: shape [N, in_dim] with same order used in training.
    returns: predicted positions [N,2]
    """
    scaler = joblib.load(scaler_path)
    Xs = scaler.transform(features.astype(np.float32))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPRegressor(in_dim=Xs.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        pred = model(torch.from_numpy(Xs).float().to(device)).cpu().numpy()
    return pred


# -----------------------------
# 5) Main
# -----------------------------
if __name__ == "__main__":
    # 1) Build dataset (replace with real ISAC/NR measurement logs)
    X, y, meta = make_synthetic_dataset(n_samples=60000, n_paths=3, seed=7)

    # Save metadata for reproducibility
    with open("ue_loc_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 2) Train
    train_model(X, y, out_prefix="ue_loc_mlp", epochs=30)

    # 3) Demo inference on a few samples
    demo_X = X[:5]
    pred_xy = load_and_predict(demo_X, "ue_loc_mlp.pt", "ue_loc_mlp_scaler.joblib")
    print("\nDemo predictions (x,y) meters:\n", pred_xy)
