"""Retrain the 32-D convolutional autoencoder on the full deduplicated dataset.

Architecture is identical to Autoencoder_v02_LD32 / v13 (Thode lab) so the
resulting checkpoint warm-starts the CNN classifier with a direct key match.

    encoder     : Conv2d(1,32,3,pad=1)+BN+ReLU+MaxPool  ×3
    to_latent   : Linear(flat,64)+ReLU+Linear(64,32)
    decoder     : symmetric reverse path
    loss        : MSE on [0,1]-normalised input
    optimiser   : Adam lr=1e-3
    scheduler   : ReduceLROnPlateau(patience=5, factor=0.5)
    early stop  : patience 10 on val reconstruction loss
    split       : grouped by date×site (same groups as CNN split)

Checkpoint saved as ``runs/ae_full/autoencoder_clean.pt`` in the same format
as the existing v13 checkpoint so ``load_pretrained_encoder`` loads it
unchanged.

Run:
    source /Users/oboulais/.../venv_bowhead/bin/activate
    PYTHONPATH=/Users/oboulais/BowheadWhaleDetector \\
    python -m bowhead.train.train_ae \\
        --data data/spectrograms_all_training.npz \\
        --out-dir runs/ae_full \\
        --epochs 100 \\
        --batch 256
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from bowhead.config import best_device
from bowhead.data.splits import grouped_split, make_date_site_group


# ---------------------------------------------------------------------------
# Autoencoder definition — mirrors Autoencoder_v02_LD32 / v13
# ---------------------------------------------------------------------------

class _Autoencoder(nn.Module):
    def __init__(self, base_channels: int = 32, latent_dim: int = 32,
                 input_hw: tuple[int, int] = (121, 104)) -> None:
        super().__init__()
        c = base_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(c, c*2, 3, padding=1), nn.BatchNorm2d(c*2), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(c*2, c*4, 3, padding=1), nn.BatchNorm2d(c*4), nn.ReLU(True), nn.MaxPool2d(2, 2),
        )

        # Infer flat dim
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_hw)
            enc_out = self.encoder(dummy)
        self._flat_dim = int(enc_out.numel())
        self._enc_spatial = enc_out.shape[-2:]   # (H', W') after conv stack

        self.to_latent = nn.Sequential(
            nn.Linear(self._flat_dim, latent_dim * 2),
            nn.ReLU(True),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # Decoder: reverse of encoder
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(True),
            nn.Linear(latent_dim * 2, self._flat_dim),
            nn.ReLU(True),
        )
        H_e, W_e = self._enc_spatial
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(c*4, c*2, 3, padding=1), nn.BatchNorm2d(c*2), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(c*2, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(c, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self._enc_channels = c * 4
        self._H_e = H_e
        self._W_e = W_e
        self._out_hw = input_hw   # (121, 104) — target size to crop/pad back to
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.to_latent(h.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = h.view(-1, self._enc_channels, self._H_e, self._W_e)
        out = self.decoder(h)
        # 3x MaxPool2d(2,2) on an odd input (121) floors each time
        # (121->60->30->15); 3x Upsample(scale_factor=2) then only gets back
        # to 120, one row short. Pad/crop back to the true input size so the
        # reconstruction always matches the target shape exactly.
        H_out, W_out = self._out_hw
        _, _, H, W = out.shape
        if H != H_out or W != W_out:
            pad_h, pad_w = H_out - H, W_out - W
            out = nn.functional.pad(
                out,
                (max(pad_w, 0) // 2, max(pad_w, 0) - max(pad_w, 0) // 2,
                 max(pad_h, 0) // 2, max(pad_h, 0) - max(pad_h, 0) // 2),
            )
            if pad_h < 0 or pad_w < 0:
                out = out[:, :, :H_out, :W_out]
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        # Align spatial dimensions: pad if recon is smaller, crop if larger.
        # (121 h → MaxPool×3 → 15 → Upsample×3 → 120, i.e. off-by-one for odd dims)
        if recon.shape[-2:] != x.shape[-2:]:
            H_t, W_t = x.shape[-2], x.shape[-1]
            H_r, W_r = recon.shape[-2], recon.shape[-1]
            if H_r < H_t or W_r < W_t:
                pad_h = max(0, H_t - H_r)
                pad_w = max(0, W_t - W_r)
                recon = torch.nn.functional.pad(recon, (0, pad_w, 0, pad_h))
            recon = recon[:, :, :H_t, :W_t]
        return recon, z


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_ae(
    data_path: Path,
    out_dir: Path,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    patience: int = 10,
    seed: int = 0,
    device: str = "auto",
    log_dir: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "auto":
        device = best_device()

    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir or str(out_dir / "tb_logs"))

    # ------------------------------------------------------------------ data
    print(f"Loading {data_path} ...")
    t0 = time.time()
    npz = np.load(data_path, allow_pickle=False)
    images = npz["images"]          # (N, H, W) uint8
    dates  = npz["date"]
    sites  = npz["site"]
    print(f"  {images.shape}  loaded in {time.time()-t0:.1f}s")

    N, H, W = images.shape

    # Grouped split by date×site — same strategy as CNN
    # AE is unsupervised; we still pass labels (from npz) to stratify the split,
    # so each partition has a similar call fraction.
    labels = npz["label"] if "label" in npz.files else np.zeros(N, dtype=np.int64)
    groups = make_date_site_group(dates, sites)
    split = grouped_split(
        labels=labels,
        groups=groups,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    idx_train, idx_val, idx_test = split.train, split.val, split.test
    print(f"  Split: train={len(idx_train):,}  val={len(idx_val):,}  test={len(idx_test):,}")

    def make_loader(idx, shuffle):
        x = torch.from_numpy(images[idx].astype(np.float32) / 255.0).unsqueeze(1)
        ds = TensorDataset(x)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(device != "cpu"))

    train_loader = make_loader(idx_train, shuffle=True)
    val_loader   = make_loader(idx_val,   shuffle=False)

    # ------------------------------------------------------------------ model
    model = _Autoencoder(base_channels=32, latent_dim=32, input_hw=(H, W)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  AE params: {n_params:,}  latent_dim={model.latent_dim}  flat={model._flat_dim}")

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=5, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()

    # ------------------------------------------------------------------ training
    best_val_loss = float("inf")
    no_improve    = 0
    history       = []

    for epoch in range(1, epochs + 1):
        # -- train
        model.train()
        train_loss = 0.0
        for (x,) in train_loader:
            x = x.to(device)
            optimiser.zero_grad()
            recon, _ = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(idx_train)

        # -- val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                recon, _ = model(x)
                val_loss += criterion(recon, x).item() * x.size(0)
        val_loss /= len(idx_val)

        scheduler.step(val_loss)
        current_lr = optimiser.param_groups[0]["lr"]

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val",   val_loss,   epoch)
        writer.add_scalar("lr",         current_lr, epoch)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            no_improve    = 0
            ckpt = {
                "epoch":        epoch,
                "val_loss":     val_loss,
                "train_loss":   train_loss,
                "encoder":      model.encoder.state_dict(),
                "to_latent":    model.to_latent.state_dict(),
                "decoder":      model.decoder.state_dict(),
                "from_latent":  model.from_latent.state_dict(),
                # "state_dict" key matches what load_pretrained_encoder expects
                "state_dict":   model.state_dict(),
                "input_hw":     [H, W],
                "latent_dim":   model.latent_dim,
                "base_channels": 32,
            }
            torch.save(ckpt, out_dir / "autoencoder_clean.pt")
        else:
            no_improve += 1

        row = {
            "epoch": epoch, "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6), "lr": current_lr, "best": is_best
        }
        history.append(row)
        flag = "*" if is_best else " "
        print(f"  [{flag}] epoch {epoch:3d}  train={train_loss:.5f}  val={val_loss:.5f}"
              f"  lr={current_lr:.2e}  no_improve={no_improve}")

        if no_improve >= patience:
            print(f"Early stop at epoch {epoch} (patience={patience})")
            break

    writer.close()

    # ------------------------------------------------------------------ results
    results = {
        "best_epoch":    next(r["epoch"] for r in history if r["best"]),
        "best_val_loss": best_val_loss,
        "n_train":       len(idx_train),
        "n_val":         len(idx_val),
        "n_test":        len(idx_test),
        "checkpoint":    str(out_dir / "autoencoder_clean.pt"),
        "history":       history,
    }

    with open(out_dir / "ae_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nBest checkpoint: {results['checkpoint']}")
    print(f"  best_epoch={results['best_epoch']}  best_val_loss={best_val_loss:.6f}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Retrain 32-D convolutional autoencoder")
    p.add_argument("--data",     type=Path, required=True,
                   help="Path to spectrograms .npz (e.g. data/spectrograms_all_training.npz)")
    p.add_argument("--out-dir",  type=Path, default=Path("runs/ae_full"),
                   help="Output directory for checkpoint and logs")
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--batch",    type=int,   default=256)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--patience", type=int,   default=10)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac",type=float, default=0.10)
    p.add_argument("--device",   default="auto")
    p.add_argument("--seed",     type=int,   default=0)
    args = p.parse_args()

    train_ae(
        data_path  = args.data,
        out_dir    = args.out_dir,
        epochs     = args.epochs,
        batch_size = args.batch,
        lr         = args.lr,
        val_frac   = args.val_frac,
        test_frac  = args.test_frac,
        patience   = args.patience,
        seed       = args.seed,
        device     = args.device,
    )


if __name__ == "__main__":
    main()
