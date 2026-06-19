"""Frozen-embedding benchmark for bowhead whale call detection and classification.

Evaluates four backbone families against classical bowhead detectors on the
8.7-million-detection Arctic dataset across decade-scale temporal drift and
strong anthropogenic noise.

Backbone families
-----------------
* Image-based (runnable NOW on spectrograms.npz):
    - AST-style ViT  (timm ViT, ImageNet pre-trained, adapted to SNR-grams)
    - ResNet-18      (ImageNet, torchvision)
    - EfficientNet-B0 (ImageNet, torchvision)
* Waveform-based (GPU cluster, raw audio required):
    - BirdNET 2.3    (TF-Hub, 48 kHz)
    - Perch 2.0      (TF-Hub, 32 kHz)
    - GMWM           (Google Multispecies Whale Model, 24 kHz)

Tasks
-----
1. **Binary detection**        call vs. non-call (Types 0 vs. 1–7)
2. **Call-type classification** Types 1–7 (seven bowhead morphological classes)

Evaluation axes
---------------
* Standard grouped few-shot (k ∈ {4,8,16,32}, 5 repeats, ROC-AUC)
* Temporal drift  — train on years {2008,2010}, test on {2012,2014}
* Site shift      — train on site 3, test on site 5 (or vice-versa)
* Open-set rejection — AUROC of max-class confidence for non-bowhead signals
* Per-call-type breakdown — fine-grained PR curve per type

Embedding fusion
----------------
Concatenation and mean-pooling across backbone combinations; includes
multi-sensor (DASAR A/D/G) context integration.

Unsupervised subdivision
------------------------
Encoder-decoder representation clustering of "complex" call types (Types 1–3)
to discover stable call subtypes without additional annotation.

Run
---
    PYTHONPATH=. python -m bowhead.benchmark.run_benchmark \\
        --data data/spectrograms.npz \\
        --backbones ast_imagenet resnet18 \\
        --out runs/benchmark_v1
"""
