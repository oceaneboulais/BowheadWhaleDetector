"""Smoke tests — no real data needed (verification steps 1-3 of the plan).

Run:  /usr/local/bin/python3.8 -m pytest tests/ -q
  or: /usr/local/bin/python3.8 tests/test_smoke.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from bowhead.models.encoder import ConvEncoder
from bowhead.models.custom_cnn import EncoderClassifier, load_pretrained_encoder
from bowhead.data.splits import grouped_split
from bowhead.eval.metrics import score_predictions, resample_to_prevalence


def test_forward_shapes_and_proba():
    model = EncoderClassifier(num_classes=2, input_hw=(121, 104))
    x = torch.randn(8, 1, 121, 104)
    logits = model(x)
    assert logits.shape == (8, 2), logits.shape
    p = model.predict_proba(x)
    assert p.shape == (8,)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0
    # encoder latent is 32-D as in the draft
    assert model.encoder(x).shape == (8, 32)


def test_warmstart_matches_ae_keys():
    """An AE-shaped checkpoint (encoder.* / to_latent.* keys) must warm-start
    every encoder tensor of the classifier, leaving only the head uninitialised."""
    # A standalone ConvEncoder has exactly the AE's encoder-path state_dict keys.
    ae_encoder = ConvEncoder(input_hw=(121, 104), latent_dim=32)
    ae_keys = set(ae_encoder.state_dict().keys())
    assert any(k.startswith("encoder.") for k in ae_keys)
    assert any(k.startswith("to_latent.") for k in ae_keys)

    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "ae.pt"
        torch.save({"state_dict": ae_encoder.state_dict()}, ckpt)

        clf = EncoderClassifier(num_classes=2, input_hw=(121, 104), latent_dim=32)
        report = load_pretrained_encoder(clf, str(ckpt))

        # every AE encoder/to_latent tensor matched, none unexpected
        assert len(report["matched"]) == len(ae_keys), report
        assert report["unexpected"] == [], report["unexpected"]
        # only the classification head is left uninitialised
        assert all(m.startswith("classifier.") for m in report["missing"]), report["missing"]

        # and the weights actually transferred (identical values)
        loaded = clf.encoder.state_dict()
        for k, v in ae_encoder.state_dict().items():
            assert torch.equal(loaded[k], v), f"weight mismatch at {k}"


def test_grouped_split_no_leakage():
    rng = np.random.default_rng(0)
    n_groups = 200
    groups_per = rng.integers(1, 5, size=n_groups)          # 1-4 detections / call
    groups = np.repeat(np.arange(n_groups), groups_per)
    # each group has a single label (call or not) -> realistic structure
    group_label = rng.integers(0, 2, size=n_groups)
    labels = np.repeat(group_label, groups_per)

    split = grouped_split(labels, groups, val_frac=0.15, test_frac=0.15, seed=0)
    tr, va, te = set(groups[split.train]), set(groups[split.val]), set(groups[split.test])
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te), "group leak!"
    # every image assigned exactly once
    total = len(split.train) + len(split.val) + len(split.test)
    assert total == len(labels)
    print(split.summary(labels, groups))


def test_metrics_sanity():
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(200), np.zeros(800)].astype(int)

    # random scores -> AUC ~ 0.5
    rand = rng.random(len(y))
    m_rand = score_predictions(y, rand)
    assert 0.4 < m_rand.roc_auc < 0.6, m_rand.roc_auc

    # perfectly-ordered scores -> AUC == 1.0
    perfect = y + rng.random(len(y)) * 0.01
    m_perfect = score_predictions(y, perfect.astype(float))
    assert m_perfect.roc_auc > 0.99, m_perfect.roc_auc

    # prevalence resampling hits ~target
    idx = resample_to_prevalence(y, target_prevalence=1 / 9, seed=0)
    prev = y[idx].mean()
    assert abs(prev - 1 / 9) < 0.03, prev


if __name__ == "__main__":
    test_forward_shapes_and_proba()
    test_warmstart_matches_ae_keys()
    test_grouped_split_no_leakage()
    test_metrics_sanity()
    print("\nALL SMOKE TESTS PASSED")
