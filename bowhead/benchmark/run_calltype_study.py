"""Run the call-type multiclass probe + unsupervised subdivision study end-to-end.

This is the analysis that backs the ML-conference companion paper (Objective 2
of the funding proposal: multiclass call-type classification and unsupervised
subdivision of morphologically ambiguous call types). It reuses:

  * ``bowhead.models.custom_cnn.EncoderClassifier`` / ``load_pretrained_encoder``
    to load the already-trained scratch/warm-start CNN encoders.
  * ``bowhead.models.encoder.ConvEncoder`` to load the autoencoder-only encoder.
  * ``bowhead.data.splits.grouped_split`` / ``make_date_site_group`` for the
    leakage-free date x site grouped split (same convention as the detector).
  * ``bowhead.benchmark.probes.call_type.evaluate_call_type`` for the 7-way
    linear probe.
  * ``bowhead.benchmark.unsupervised.cluster.fit_clusters`` /
    ``bic_select_gmm`` for k-means / GMM subdivision of the call-type
    embedding space, scored against the manual Type 1-7 labels with ARI/NMI.

Run:
    PYTHONPATH=. python -m bowhead.benchmark.run_calltype_study \\
        --data data/spectrograms_100k_matched.npz \\
        --ae runs/ae_100k_matched/autoencoder_clean.pt \\
        --scratch runs/scratch_100k_matched/best.pt \\
        --warmstart runs/warmstart_100k_matched/best.pt \\
        --out runs/calltype_study
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from bowhead.config import best_device
from bowhead.data.dataset import per_sample_minmax
from bowhead.data.splits import grouped_split, make_date_site_group
from bowhead.models.custom_cnn import EncoderClassifier, load_pretrained_encoder
from bowhead.models.encoder import ConvEncoder
from bowhead.benchmark.probes.call_type import evaluate_call_type
from bowhead.benchmark.unsupervised.cluster import fit_clusters, bic_select_gmm


# ── encoder loading ──────────────────────────────────────────────────────────

def _load_ae_encoder(ckpt_path: str, device: str) -> ConvEncoder:
    ck = torch.load(ckpt_path, map_location=device)
    enc = ConvEncoder(
        in_channels=1,
        input_hw=tuple(ck["input_hw"]),
        base_channels=ck["base_channels"],
        latent_dim=ck["latent_dim"],
    ).to(device)
    state = ck["model_state"]
    enc_state = {
        k: v for k, v in state.items() if k.startswith("encoder.") or k.startswith("to_latent.")
    }
    missing = enc.load_state_dict(enc_state, strict=True)
    enc.eval()
    return enc


def _load_cnn_encoder(ckpt_path: str, device: str) -> EncoderClassifier:
    model = EncoderClassifier(input_hw=(121, 104)).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    state = ck.get("state_dict", ck)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def _extract(encoder: nn.Module, images: np.ndarray, device: str, batch: int = 512) -> np.ndarray:
    """Return (N, D) float32 embeddings from any of the three encoder types."""
    out = []
    for start in range(0, len(images), batch):
        chunk = images[start : start + batch]
        x = np.stack([per_sample_minmax(im) for im in chunk])[:, None].astype(np.float32)
        t = torch.from_numpy(x).to(device)
        z = encoder(t)
        out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0)


# ── study ─────────────────────────────────────────────────────────────────────

def run(
    data_path: str,
    ae_ckpt: str,
    scratch_ckpt: str,
    warmstart_ckpt: str,
    out_dir: str,
    device: str,
    seed: int = 0,
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if device == "auto":
        device = best_device()
    print(f"Device: {device}")

    print(f"Loading {data_path} ...")
    data = np.load(data_path, allow_pickle=True)
    images = data["images"]
    labels = data["label"].astype(int)
    call_type = data["call_type"].astype(int)
    date = data["date"]
    site = data["site"]
    groups = make_date_site_group(date, site)
    print(f"  {len(images):,} images; {int((labels==1).sum()):,} calls "
          f"across types {sorted(np.unique(call_type[call_type>0]).tolist())}")

    # ── leakage-free grouped split (identical convention to the detector) ────
    split = grouped_split(labels, groups, val_frac=0.0, test_frac=0.15, seed=seed)
    train_idx, test_idx = split.train, split.test
    print(split.summary(labels, groups))

    # Restrict to positives (call types 1-7) for the call-type task.
    train_pos = train_idx[call_type[train_idx] > 0]
    test_pos = test_idx[call_type[test_idx] > 0]
    print(f"  call-type task: {len(train_pos):,} train / {len(test_pos):,} test positives")

    encoders = {
        "autoencoder": _load_ae_encoder(ae_ckpt, device),
        "cnn_scratch": _load_cnn_encoder(scratch_ckpt, device).encoder,
        "cnn_warmstart": _load_cnn_encoder(warmstart_ckpt, device).encoder,
    }

    results: dict = {"call_type_probe": {}, "clustering": {}}

    for name, enc in encoders.items():
        print(f"\n== {name} ==")
        emb_train_pos = _extract(enc, images[train_pos], device)
        emb_test_pos = _extract(enc, images[test_pos], device)

        # 7-way linear probe -----------------------------------------------
        res = evaluate_call_type(
            emb_train_pos, call_type[train_pos],
            emb_test_pos, call_type[test_pos],
            backbone_name=name, seed=seed,
        )
        print(res.summary())
        results["call_type_probe"][name] = {
            "roc_auc_macro": res.roc_auc_macro,
            "accuracy": res.accuracy,
            "per_type_roc_auc": res.per_type_roc_auc,
            "n_train": int(len(train_pos)),
            "n_test": int(len(test_pos)),
        }

        # unsupervised subdivision on ALL positives (train+test embeddings) --
        emb_all_pos = np.concatenate([emb_train_pos, emb_test_pos], axis=0)
        type_all_pos = np.concatenate([call_type[train_pos], call_type[test_pos]])

        best_k, bics = bic_select_gmm(emb_all_pos, k_range=(2, 4, 6, 7, 8, 10, 12, 16), seed=seed)
        km = fit_clusters(emb_all_pos, method="k_means", n_clusters=7, seed=seed)
        gmm_best = fit_clusters(emb_all_pos, method="gmm", n_clusters=best_k, seed=seed)

        sil_km = float(silhouette_score(emb_all_pos, km.labels)) if km.n_clusters > 1 else float("nan")
        sil_gmm = float(silhouette_score(emb_all_pos, gmm_best.labels)) if gmm_best.n_clusters > 1 else float("nan")

        ari_km = float(adjusted_rand_score(type_all_pos, km.labels))
        nmi_km = float(normalized_mutual_info_score(type_all_pos, km.labels))
        ari_gmm = float(adjusted_rand_score(type_all_pos, gmm_best.labels))
        nmi_gmm = float(normalized_mutual_info_score(type_all_pos, gmm_best.labels))

        print(f"  k-means(k=7)   silhouette={sil_km:.3f}  ARI={ari_km:.3f}  NMI={nmi_km:.3f}")
        print(f"  GMM(BIC k={best_k})  silhouette={sil_gmm:.3f}  ARI={ari_gmm:.3f}  NMI={nmi_gmm:.3f}")

        results["clustering"][name] = {
            "kmeans_k7": {"silhouette": sil_km, "ari": ari_km, "nmi": nmi_km},
            "gmm_bic": {"k": int(best_k), "bics": bics, "silhouette": sil_gmm,
                        "ari": ari_gmm, "nmi": nmi_gmm},
            "n_positives": int(len(emb_all_pos)),
        }

        np.savez_compressed(
            Path(out_dir) / f"embeddings_{name}.npz",
            emb=emb_all_pos, call_type=type_all_pos,
            date=np.concatenate([date[train_pos], date[test_pos]]),
            site=np.concatenate([site[train_pos], site[test_pos]]),
        )

    out_path = Path(out_dir) / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWritten {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/spectrograms_100k_matched.npz")
    p.add_argument("--ae", default="runs/ae_100k_matched/autoencoder_clean.pt")
    p.add_argument("--scratch", default="runs/scratch_100k_matched/best.pt")
    p.add_argument("--warmstart", default="runs/warmstart_100k_matched/best.pt")
    p.add_argument("--out", default="runs/calltype_study")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    run(a.data, a.ae, a.scratch, a.warmstart, a.out, a.device, a.seed)
