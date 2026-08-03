"""Train the custom CNN (Approach 1): AE encoder trunk + softmax head.

Pipeline:
    load data -> grouped split -> (optional) warm-start encoder ->
    train w/ early stopping on val ROC-AUC -> evaluate on the held-out test set
    at realistic prevalence.

Run:
    python -m bowhead.train.train_cnn --data data/spectrograms.npz \
        --warm-start runs/ae_v13.pt --tag warmstart
    python -m bowhead.train.train_cnn --data data/spectrograms.npz --tag scratch
The two runs above ARE the warm-start vs random-init ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

from bowhead.config import TrainConfig, best_device
from bowhead.data.dataset import SpectrogramDataset
from bowhead.data.splits import grouped_split, make_date_site_group
from bowhead.models.custom_cnn import EncoderClassifier, load_pretrained_encoder
from bowhead.scorers import CNNScorer
from bowhead.eval.evaluate import evaluate_scorer, metrics_table
from bowhead.train.augment import spec_augment, mixup_batch
from bowhead.train.losses import FocalLoss


def _build_groups(metadata: dict, group_col: str) -> np.ndarray:
    if group_col == "date_site":
        return make_date_site_group(metadata["date"], metadata["site"])
    if group_col in metadata:
        return np.asarray(metadata[group_col])
    raise KeyError(f"group_col {group_col!r} not in metadata keys {list(metadata)}")


@torch.no_grad()
def _val_auc(model: EncoderClassifier, loader: DataLoader, device: str) -> float:
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        p = model.predict_proba(x.to(device)).cpu().numpy()
        scores.append(p)
        labels.append(y.numpy())
    return float(roc_auc_score(np.concatenate(labels), np.concatenate(scores)))


def train_custom_cnn(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = best_device() if cfg.device == "auto" else cfg.device
    out = Path(cfg.out_dir) / cfg.tag
    out.mkdir(parents=True, exist_ok=True)
    # TensorBoard: event files land in runs/<tag>/ so `tensorboard --logdir runs`
    # shows every run (and the warm-start vs scratch arms) side by side.
    writer = SummaryWriter(log_dir=str(out))

    # --- data + leakage-free split -------------------------------------- #
    images, labels, metadata = SpectrogramDataset.load_npz(cfg.data_path)

    # Auto-detect channel count from the loaded array so 1-ch and 2-ch npz
    # files work without any extra CLI flag.
    auto_channels = images.shape[1] if images.ndim == 4 else 1
    if auto_channels != cfg.in_channels:
        print(f"Auto-detected in_channels={auto_channels} from data "
              f"(config had {cfg.in_channels}); overriding.")
        cfg.in_channels = auto_channels

    groups = _build_groups(metadata, cfg.group_col)
    split = grouped_split(
        labels, groups, val_frac=cfg.val_frac, test_frac=cfg.test_frac, seed=cfg.seed
    )
    print("Grouped split (verify 0 overlap):")
    print(split.summary(labels, groups))

    train_ds = SpectrogramDataset(images, labels, split.train)
    val_ds = SpectrogramDataset(images, labels, split.val)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    # --- model (+ optional warm-start) ---------------------------------- #
    model = EncoderClassifier(
        num_classes=cfg.num_classes, in_channels=cfg.in_channels,
        input_hw=cfg.input_hw, latent_dim=cfg.latent_dim, dropout=cfg.dropout,
    ).to(device)
    if cfg.warm_start_ckpt:
        report = load_pretrained_encoder(
            model, cfg.warm_start_ckpt, source_prefix=cfg.warm_start_source_prefix
        )
        print(f"Warm-start: matched {len(report['matched'])} encoder tensors, "
              f"missing {len(report['missing'])}, unexpected {len(report['unexpected'])}")
    if cfg.freeze_encoder:
        model.freeze_encoder(True)

    # class-weighted loss to handle residual imbalance in the (grouped) train set
    if cfg.class_weighted_loss:
        train_labels = labels[split.train]
        counts = np.bincount(train_labels, minlength=cfg.num_classes).astype(float)
        weights = torch.tensor(counts.sum() / (cfg.num_classes * np.maximum(counts, 1)),
                               dtype=torch.float32, device=device)
    else:
        weights = None

    if cfg.use_focal_loss:
        criterion = FocalLoss(
            gamma=cfg.focal_gamma,
            alpha=cfg.focal_alpha,
            weight=weights,
        )
        print(f"Loss: FocalLoss(gamma={cfg.focal_gamma}, alpha={cfg.focal_alpha})")
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("Loss: CrossEntropyLoss")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.use_spec_augment:
        print(f"SpecAugment: F={cfg.spec_aug_F}, T={cfg.spec_aug_T}, "
              f"n_freq={cfg.spec_aug_n_freq}, n_time={cfg.spec_aug_n_time}")
    if cfg.use_mixup:
        print(f"Mixup: alpha={cfg.mixup_alpha}")

    # --- training loop w/ early stopping on val AUC --------------------- #
    best_auc, best_epoch, since_improve = -1.0, -1, 0
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            if cfg.use_spec_augment:
                x = spec_augment(
                    x,
                    F=cfg.spec_aug_F,
                    T=cfg.spec_aug_T,
                    n_freq_masks=cfg.spec_aug_n_freq,
                    n_time_masks=cfg.spec_aug_n_time,
                )
            if cfg.use_mixup:
                x, y = mixup_batch(x, y, alpha=cfg.mixup_alpha)
            elif cfg.use_focal_loss:
                # FocalLoss handles int64 targets when no mixup
                pass
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(x)
        train_loss = running / len(train_ds)
        auc = _val_auc(model, val_dl, device)
        print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_auc {auc:.4f}")
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("auc/val", auc, epoch)

        if auc > best_auc:
            best_auc, best_epoch, since_improve = auc, epoch, 0
            torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg),
                        "val_auc": auc, "epoch": epoch}, out / "best.pt")
        else:
            since_improve += 1
            if since_improve >= cfg.early_stop_patience:
                print(f"early stop at epoch {epoch} (best {best_auc:.4f} @ {best_epoch})")
                break

    # --- final test-set evaluation at realistic prevalence -------------- #
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["state_dict"])
    test_images = images[split.test]
    test_labels = labels[split.test]
    scorer = CNNScorer(model, name=cfg.tag, device=device)
    test_metrics = evaluate_scorer(
        scorer, test_images, test_labels,
        target_prevalence=cfg.eval_prevalence, seed=cfg.seed,
    )
    print("\nHeld-out test (realistic prevalence):")
    print(metrics_table({cfg.tag: test_metrics}))

    # Log final test metrics + the run's hyperparameters so the TensorBoard
    # HPARAMS tab compares warm-start vs scratch arms at a glance.
    test_scalars = test_metrics.scalar_dict()
    p_at_r70 = test_metrics.precision_at_recall(0.70)
    for k, v in test_scalars.items():
        writer.add_scalar(f"test/{k}", v, best_epoch)
    writer.add_scalar("test/precision_at_recall_0.70", p_at_r70, best_epoch)
    writer.add_hparams(
        {"tag": cfg.tag, "warm_start": bool(cfg.warm_start_ckpt),
         "freeze_encoder": cfg.freeze_encoder, "lr": cfg.lr,
         "batch_size": cfg.batch_size, "latent_dim": cfg.latent_dim,
         "group_col": cfg.group_col, "seed": cfg.seed},
        {"hparam/best_val_auc": best_auc,
         "hparam/test_roc_auc": test_scalars.get("roc_auc", float("nan")),
         "hparam/test_avg_precision": test_scalars.get("average_precision", float("nan")),
         "hparam/test_precision_at_recall_0.70": p_at_r70},
    )
    writer.close()

    summary = {
        "tag": cfg.tag,
        "best_val_auc": best_auc,
        "best_epoch": best_epoch,
        "test": test_metrics.scalar_dict(),
        "test_precision_at_recall_0.70": test_metrics.precision_at_recall(0.70),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train custom CNN (Approach 1)")
    p.add_argument("--data", dest="data_path", default=TrainConfig.data_path)
    p.add_argument("--warm-start", dest="warm_start_ckpt", default=None)
    p.add_argument("--group-col", dest="group_col", default=TrainConfig.group_col)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--device", default=TrainConfig.device)
    p.add_argument("--tag", default=TrainConfig.tag)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    # augmentation
    p.add_argument("--spec-augment", dest="use_spec_augment", action="store_true")
    p.add_argument("--spec-aug-F", type=int, default=TrainConfig.spec_aug_F)
    p.add_argument("--spec-aug-T", type=int, default=TrainConfig.spec_aug_T)
    p.add_argument("--spec-aug-n-freq", type=int, default=TrainConfig.spec_aug_n_freq)
    p.add_argument("--spec-aug-n-time", type=int, default=TrainConfig.spec_aug_n_time)
    p.add_argument("--mixup", dest="use_mixup", action="store_true")
    p.add_argument("--mixup-alpha", type=float, default=TrainConfig.mixup_alpha)
    # loss
    p.add_argument("--focal-loss", dest="use_focal_loss", action="store_true")
    p.add_argument("--focal-gamma", type=float, default=TrainConfig.focal_gamma)
    p.add_argument("--focal-alpha", type=float, default=None)
    a = p.parse_args()
    return TrainConfig(**vars(a))


if __name__ == "__main__":
    train_custom_cnn(_parse_args())
