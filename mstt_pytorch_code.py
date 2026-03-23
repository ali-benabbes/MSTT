import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Configuration
# ============================================================

@dataclass
class MSTTConfig:
    num_nodes: int = 128            # number of spatial grid cells / pixels
    in_features: int = 3            # NDVI, LST, Soil Moisture
    seq_len: int = 12               # past monthly observations
    pred_len: int = 1               # next-month SPI
    d_model: int = 64
    num_heads: int = 8
    ff_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 20
    patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, D)
        t = x.size(1)
        return x + self.pe[:, :t].unsqueeze(2)


# ============================================================
# Core MSTT Blocks
# ============================================================

class SpatialAttentionBlock(nn.Module):
    """
    Applies self-attention across spatial nodes for each time step.
    Input:  (B, T, N, D)
    Output: (B, T, N, D)
    """
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, d = x.shape
        x_reshaped = x.reshape(b * t, n, d)
        attn_out, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)
        x_reshaped = self.norm1(x_reshaped + self.dropout(attn_out))
        ff_out = self.ff(x_reshaped)
        x_reshaped = self.norm2(x_reshaped + self.dropout(ff_out))
        return x_reshaped.reshape(b, t, n, d)


class TemporalAttentionBlock(nn.Module):
    """
    Applies self-attention across time for each node.
    Input:  (B, T, N, D)
    Output: (B, T, N, D)
    """
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, d = x.shape
        x_perm = x.permute(0, 2, 1, 3).reshape(b * n, t, d)
        attn_out, _ = self.attn(x_perm, x_perm, x_perm)
        x_perm = self.norm1(x_perm + self.dropout(attn_out))
        ff_out = self.ff(x_perm)
        x_perm = self.norm2(x_perm + self.dropout(ff_out))
        return x_perm.reshape(b, n, t, d).permute(0, 2, 1, 3)


class MSTTLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.spatial_block = SpatialAttentionBlock(d_model, num_heads, ff_dim, dropout)
        self.temporal_block = TemporalAttentionBlock(d_model, num_heads, ff_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_block(x)
        x = self.temporal_block(x)
        return x


# ============================================================
# Full MSTT Model
# ============================================================

class MSTT(nn.Module):
    """
    Multivariate Spatio-Temporal Transformer for grid-based SPI forecasting.

    Input:
        x: (B, T, N, F)
    Output:
        y_hat: (B, pred_len, N)
    """
    def __init__(self, config: MSTTConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.in_features, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, max_len=config.seq_len + 8)
        self.layers = nn.ModuleList([
            MSTTLayer(config.d_model, config.num_heads, config.ff_dim, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, T, N, F), got {tuple(x.shape)}")

        x = self.input_proj(x)                   # (B, T, N, D)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        # use the last encoded time step to predict next SPI
        last_state = x[:, -1, :, :]            # (B, N, D)
        y_hat = self.head(last_state)          # (B, N, pred_len)
        y_hat = y_hat.permute(0, 2, 1)         # (B, pred_len, N)
        return y_hat


# ============================================================
# Dataset
# ============================================================

class SpatioTemporalDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        x: (samples, T, N, F)
        y: (samples, pred_len, N)
        """
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


# ============================================================
# Metrics
# ============================================================

def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def r2_score_torch(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    if ss_tot == 0:
        return 0.0
    return (1 - ss_res / ss_tot).item()


# ============================================================
# Training Utilities
# ============================================================

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: str) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        total_loss += loss.item() * xb.size(0)
        all_preds.append(pred)
        all_targets.append(yb)

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    return {
        "loss": total_loss / len(loader.dataset),
        "rmse": rmse(preds, targets),
        "mae": mae(preds, targets),
        "r2": r2_score_torch(preds, targets),
    }


class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience


# ============================================================
# Example Synthetic Data Generator
# Replace this with your UAE EO cube loader.
# ============================================================

def generate_synthetic_data(samples: int, seq_len: int, num_nodes: int, in_features: int,
                            pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.rand(samples, seq_len, num_nodes, in_features).astype(np.float32)

    # synthetic SPI target linked to last-step NDVI, LST, SM
    last_step = x[:, -1, :, :]
    ndvi = last_step[:, :, 0]
    lst = last_step[:, :, 1]
    sm = last_step[:, :, 2]

    spi = 0.5 * ndvi - 0.25 * lst + 0.75 * sm
    spi = (spi - spi.mean()) / (spi.std() + 1e-8)
    y = np.expand_dims(spi, axis=1).astype(np.float32)  # (samples, 1, N)

    return x, y


# ============================================================
# Main Training Script
# ============================================================

def main() -> None:
    config = MSTTConfig()
    set_seed(config.seed)
    device = config.device
    print(f"Using device: {device}")

    x, y = generate_synthetic_data(
        samples=320,
        seq_len=config.seq_len,
        num_nodes=config.num_nodes,
        in_features=config.in_features,
        pred_len=config.pred_len,
    )

    n_total = len(x)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
    x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]

    train_loader = DataLoader(SpatioTemporalDataset(x_train, y_train), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(SpatioTemporalDataset(x_val, y_val), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(SpatioTemporalDataset(x_test, y_test), batch_size=config.batch_size, shuffle=False)

    model = MSTT(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    stopper = EarlyStopping(patience=config.patience)

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f} | "
            f"val_mae={val_metrics['mae']:.4f} | "
            f"val_r2={val_metrics['r2']:.4f}"
        )

        if stopper.step(val_metrics["loss"], model):
            print("Early stopping triggered.")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    test_metrics = evaluate(model, test_loader, criterion, device)
    print("\nTest metrics")
    print(f"Loss : {test_metrics['loss']:.4f}")
    print(f"RMSE : {test_metrics['rmse']:.4f}")
    print(f"MAE  : {test_metrics['mae']:.4f}")
    print(f"R2   : {test_metrics['r2']:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/mstt_model.pt")
    print("\nSaved model to checkpoints/mstt_model.pt")


if __name__ == "__main__":
    main()
