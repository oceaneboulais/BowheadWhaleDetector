"""Approach 1: supervised CNN reusing the autoencoder encoder trunk.

The encoder trunk (3x Conv+BN+ReLU+MaxPool, FC-64, FC-32) is kept intact and a
small classification head is placed on the 32-D latent, trained end-to-end with
cross-entropy. Two ablation modes:

    * warm-start  -- initialise the trunk from the trained autoencoder encoder
    * random init -- train the same architecture from scratch

The difference isolates how much the unsupervised features actually help.
"""

from __future__ import annotations

import torch
from torch import nn

from bowhead.models.encoder import ConvEncoder


class EncoderClassifier(nn.Module):
    """Encoder trunk + linear classification head over the latent vector."""

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        input_hw: tuple[int, int] = (121, 104),
        latent_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(
            in_channels=in_channels, input_hw=input_hw, latent_dim=latent_dim
        )
        head: list[nn.Module] = []
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(latent_dim, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (use with ``nn.CrossEntropyLoss``)."""
        z = self.encoder(x)
        return self.classifier(z)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Call-probability for the positive (call) class, shape (N,)."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)[:, 1]

    def freeze_encoder(self, freeze: bool = True) -> None:
        """Optionally freeze the trunk (e.g. linear-probe style training)."""
        for p in self.encoder.parameters():
            p.requires_grad = not freeze


# The two Sequential blocks that make up the autoencoder's encoder path. Both
# are warm-started: the conv trunk AND the FC bottleneck (FC-64, FC-32), per the
# "reuse the encoder trunk incl. FC-64/FC-32" spec.
_AE_ENCODER_BLOCKS = ("encoder.", "to_latent.")


def load_pretrained_encoder(
    model: EncoderClassifier,
    checkpoint_path: str,
    source_prefix: str = "",
    map_location: str = "cpu",
) -> dict[str, list[str]]:
    """Warm-start the classifier's encoder trunk from an autoencoder checkpoint.

    The AE checkpoint stores the encoder path under top-level keys ``encoder.*``
    and ``to_latent.*`` (see ``bowhead/models/encoder.py``). In the classifier
    the trunk is the ``encoder`` submodule (a ``ConvEncoder``), so those keys map
    to ``encoder.encoder.*`` / ``encoder.to_latent.*``. We remap and load with
    ``strict=False``; the classification head is absent from the AE checkpoint
    and so stays randomly initialised (it appears in ``missing``).

    ``source_prefix`` strips any wrapper the checkpoint nests the model under
    (e.g. ``"model."``). Returns ``matched`` / ``missing`` / ``unexpected`` keys.
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if source_prefix:
        state = {
            k[len(source_prefix):]: v
            for k, v in state.items()
            if k.startswith(source_prefix)
        }

    # Remap AE encoder-path keys onto the classifier's nested `encoder` trunk.
    remapped = {
        f"encoder.{k}": v
        for k, v in state.items()
        if any(k.startswith(b) for b in _AE_ENCODER_BLOCKS)
    }
    if not remapped:
        raise ValueError(
            f"No encoder-path keys ({_AE_ENCODER_BLOCKS}) found in checkpoint. "
            f"Top-level prefixes present: {sorted({k.split('.')[0] for k in state})}"
        )

    result = model.load_state_dict(remapped, strict=False)
    return {
        "matched": [k for k in remapped if k not in result.unexpected_keys],
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
    }
