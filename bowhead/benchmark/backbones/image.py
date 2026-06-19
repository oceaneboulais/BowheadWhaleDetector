"""Image-based frozen-embedding backbones.

These backbones accept spectrogram IMAGES (N, H, W) and return (N, D) embeddings.
They run immediately on ``spectrograms.npz`` — no raw audio or TF required.

Backbones
---------
ast_imagenet   AudioSpectrogram-Transformer-style ViT, pretrained on ImageNet-21k
               via timm (vit_base_patch16_224). Spectrogram patches treated as
               image patches; gives the closest architecture match to the real AST
               (Gong et al. 2021) while being runnable without the AST audio
               checkpoint. Swap in the real AST weights when available on the cluster.

resnet18       ResNet-18, ImageNet-1k (torchvision). 512-D global-avg-pool embedding.

efficientnet_b0  EfficientNet-B0, ImageNet-1k (torchvision). 1280-D embedding.

All backbones share a common ``ImageBackbone`` interface:
    .name   str
    .dim    int    embedding dimensionality
    .embed(images: np.ndarray) -> np.ndarray   # (N, H, W) uint8 → (N, D) float32
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm

from bowhead.config import best_device
from bowhead.data.dataset import per_sample_minmax


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_3ch_float(images: np.ndarray) -> torch.Tensor:
    """(N, H, W) uint8 -> (N, 3, 224, 224) float32, per-sample min-max normalised.

    All image backbones were pretrained on 3-channel 224×224 inputs; we replicate
    the single-channel SNR-gram across all three channels (no information loss,
    no channel-mean mismatch) and bilinearly resize to 224×224.
    """
    import torch.nn.functional as F

    n = len(images)
    out = np.empty((n, 1, images.shape[1], images.shape[2]), dtype=np.float32)
    for i, img in enumerate(images):
        out[i, 0] = per_sample_minmax(img)
    t = torch.from_numpy(out)
    # replicate channel and resize
    t = t.expand(-1, 3, -1, -1)
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    # ImageNet normalisation
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (t - mean) / std


class ImageBackbone:
    """Shared interface for all image-based frozen-embedding backbones."""

    name: str
    dim: int

    def __init__(self, model: nn.Module, dim: int, name: str, device: str) -> None:
        self.model  = model.to(device).eval()
        self.dim    = dim
        self.name   = name
        self.device = device

    @torch.no_grad()
    def embed(self, images: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """(N, H, W) uint8 → (N, D) float32 embeddings."""
        all_embs = []
        for start in range(0, len(images), batch_size):
            batch = _to_3ch_float(images[start:start + batch_size]).to(self.device)
            emb = self.model(batch)
            # ViT models may return (B, tokens, D) — take the CLS token
            if emb.ndim == 3:
                emb = emb[:, 0, :]
            all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0)


# ── factory functions ─────────────────────────────────────────────────────────

def make_ast_imagenet(device: str | None = None) -> ImageBackbone:
    """ViT-Base/16 pretrained on ImageNet-21k (timm) — AST-style architecture.

    This is the architecture of the Audio Spectrogram Transformer (Gong et al.
    2021); the weights here are ImageNet-pretrained rather than AudioSet-pretrained.
    To use the real AST weights (AudioSet/Speech), swap in the timm checkpoint:
        timm.create_model('ast', pretrained=True)   # requires timm >= 0.9
    or load from HuggingFace: 'MIT/ast-finetuned-audioset-10-10-0.4593'.
    """
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "timm is required for the AST backbone. Install with: pip install timm"
        ) from e

    device = device or best_device()
    model = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=0  # num_classes=0 → embedding
    )
    dim = model.embed_dim  # 768 for ViT-Base
    return ImageBackbone(model, dim=dim, name="ast_imagenet", device=device)


def make_resnet18(device: str | None = None) -> ImageBackbone:
    device = device or best_device()
    m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    return ImageBackbone(m, dim=512, name="resnet18", device=device)


def make_efficientnet_b0(device: str | None = None) -> ImageBackbone:
    device = device or best_device()
    m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier = nn.Identity()
    return ImageBackbone(m, dim=1280, name="efficientnet_b0", device=device)


#: Registry — add new image backbones here
IMAGE_BACKBONE_REGISTRY: dict[str, callable] = {
    "ast_imagenet":    make_ast_imagenet,
    "resnet18":        make_resnet18,
    "efficientnet_b0": make_efficientnet_b0,
}


def load_image_backbone(name: str, device: str | None = None) -> ImageBackbone:
    """Load an image backbone by name.  Raises KeyError for unknown names."""
    if name not in IMAGE_BACKBONE_REGISTRY:
        raise KeyError(
            f"Unknown image backbone {name!r}. "
            f"Available: {list(IMAGE_BACKBONE_REGISTRY)}"
        )
    return IMAGE_BACKBONE_REGISTRY[name](device)
