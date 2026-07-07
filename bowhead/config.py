"""Experiment configuration.

Defaults follow the autoencoder training described in the JASA draft (Adam,
lr=1e-3, batch 32) so the supervised CNN trains under comparable conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def best_device() -> str:
    """Return 'mps', 'cuda', or 'cpu' — whichever is available first."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class TrainConfig:
    # data
    data_path: str = "data/spectrograms.npz"
    group_col: str = "unique_call"        # or "date_site" for a coarser split
    val_frac: float = 0.15
    test_frac: float = 0.15
    input_hw: tuple[int, int] = (121, 104)
    in_channels: int = 1

    # model
    num_classes: int = 2                  # binary call / non-call first
    latent_dim: int = 32
    dropout: float = 0.0
    warm_start_ckpt: str | None = None    # path to trained AE ckpt, or None
    warm_start_source_prefix: str = ""    # strip any wrapper (e.g. "model.")
    freeze_encoder: bool = False          # True -> linear-probe style

    # optimization (matches the AE: Adam, lr=1e-3, batch 32)
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    class_weighted_loss: bool = True      # guards against residual imbalance
    early_stop_patience: int = 10         # epochs w/o val-AUC improvement

    # augmentation (Tier-1 AP boosters)
    use_spec_augment: bool = False        # SpecAugment freq+time masking
    spec_aug_F: int = 15                  # max frequency-mask width (bins)
    spec_aug_T: int = 12                  # max time-mask width (bins)
    spec_aug_n_freq: int = 1              # number of frequency masks
    spec_aug_n_time: int = 1              # number of time masks
    use_mixup: bool = False               # Mixup interpolation
    mixup_alpha: float = 0.2             # Beta distribution parameter

    # loss function
    use_focal_loss: bool = False          # replace CE with Focal loss
    focal_gamma: float = 2.0             # focusing parameter (0 = CE)
    focal_alpha: float | None = None     # positive-class prior weight

    # eval
    eval_prevalence: float = 1.0 / 9.0    # ~1 call : 8 transients

    # bookkeeping
    seed: int = 0
    out_dir: str = "runs"
    device: str = "auto"                  # "auto" -> mps > cuda > cpu at runtime
    tag: str = "custom_cnn"

    metadata: dict = field(default_factory=dict)
