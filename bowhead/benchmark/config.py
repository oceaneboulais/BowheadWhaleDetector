"""Benchmark configuration.

Separates what can run NOW (image backbones on spectrograms.npz) from what
needs the GPU cluster and raw audio (waveform backbones).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Backbone registries ──────────────────────────────────────────────────────

#: Image backbones that operate on spectrogram images (runnable on spectrograms.npz)
IMAGE_BACKBONES = ("ast_imagenet", "resnet18", "efficientnet_b0")

#: Waveform backbones requiring raw audio + TF cluster environment
WAVEFORM_BACKBONES = ("birdnet", "perch", "gmwm")

#: All four backbone families described in the proposal
ALL_BACKBONES = IMAGE_BACKBONES + WAVEFORM_BACKBONES

#: Call-type labels used in the multiclass task (Type 0 = non-call, excluded)
CALL_TYPES = (1, 2, 3, 4, 5, 6, 7)

#: Years present in the dataset — used for temporal-drift splits
DATASET_YEARS = ("2008", "2010", "2012", "2014")

#: Sites present in the dataset — used for site-shift splits
DATASET_SITES = ("3", "5")

#: DASAR instruments — used for multi-sensor fusion
DATASET_DASARS = ("A", "D", "G")


@dataclass
class BenchmarkConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_path: str = "data/spectrograms.npz"
    group_col: str = "date_site"          # leakage-free split key

    # ── Backbones to evaluate ─────────────────────────────────────────────────
    #: Image backbones (run locally; subset of IMAGE_BACKBONES)
    image_backbones: tuple[str, ...] = IMAGE_BACKBONES
    #: Waveform backbones (cluster-only; leave empty for local runs)
    waveform_backbones: tuple[str, ...] = ()

    # ── Embedding fusion combinations ─────────────────────────────────────────
    #: Each inner tuple names the backbones to concatenate for a fused probe.
    #: e.g. [("resnet18", "ast_imagenet"), ("resnet18", "ast_imagenet", "perch")]
    fusion_combinations: tuple[tuple[str, ...], ...] = (
        ("resnet18", "ast_imagenet"),
    )

    # ── Probe hyper-parameters (mirror Burns et al. 2025) ────────────────────
    few_shot_k: tuple[int, ...] = (4, 8, 16, 32)
    few_shot_repeats: int = 5
    probe_C: float = 1.0               # logistic-regression regularisation
    probe_max_iter: int = 1000

    # ── Evaluation axes ───────────────────────────────────────────────────────
    eval_temporal_drift: bool = True
    eval_site_shift: bool = True
    eval_open_set: bool = True
    eval_call_type: bool = True         # multiclass call-type task

    # ── Temporal split ────────────────────────────────────────────────────────
    #: Train on these years, test on the rest
    temporal_train_years: tuple[str, ...] = ("2008", "2010")
    temporal_test_years: tuple[str, ...] = ("2012", "2014")

    # ── Site-shift split ──────────────────────────────────────────────────────
    site_train: str = "3"
    site_test: str = "5"

    # ── Call types for multiclass task ────────────────────────────────────────
    call_types: tuple[int, ...] = CALL_TYPES

    # ── Multi-sensor fusion ───────────────────────────────────────────────────
    eval_sensor_fusion: bool = True    # fuse A/D/G embeddings per detection event

    # ── Unsupervised subdivision ──────────────────────────────────────────────
    run_subdivision: bool = False      # expensive; off by default
    #: Which call types to subdivide (the morphologically "complex" ones)
    subdivision_types: tuple[int, ...] = (1, 2, 3)
    subdivision_n_clusters: int = 5    # per complex type

    # ── Bookkeeping ───────────────────────────────────────────────────────────
    out_dir: str = "runs/benchmark"
    seed: int = 0
    device: str = "auto"               # auto → mps > cuda > cpu
    batch_size: int = 256
