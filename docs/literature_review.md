# Literature Review — CNN-based whale call detection & transfer learning

*Working draft for "Enhancing bowhead whale detection using convolutional autoencoders" (Thode 2026, JASA). First pass assembled 2026-05-29. Citations to be cross-checked against `Bowhead_AI_references.bib` and completed with full bibliographic detail before submission.*

## 1. Purpose
Frame the contribution of (a) the unsupervised convolutional autoencoder + kNN detector and (b) the three supervised CNN benchmarks (custom encoder-head CNN, BirdNET transfer, Perch 2.0 transfer) against the state of the art in deep-learning whale-call detection.

## 2. The shift from hand-crafted features to end-to-end CNNs
Early automated detectors — including the project's own original detector (Thode et al. 2012) — used image-processing to build hand-crafted feature vectors fed to a shallow (≈3-layer) neural network; these could detect simple frequency-modulated calls but not classify them, and struggled with complex calls and airgun interference. The field has since moved to CNNs that learn features end-to-end from spectrogram images.

Key demonstrations that CNNs outperform classical/hand-crafted detectors:
- **Shiu et al. (2020), *Scientific Reports*** — "Deep neural networks for automated detection of marine mammal species." Canonical CNN detector for North Atlantic right whale (NARW) upcalls on DCLDE 2013 data. CNNs had higher precision/recall and far fewer false positives than contemporary methods: <0.30 false positives/hour at 0.70 recall. https://www.nature.com/articles/s41598-020-57549-y
- **Allen et al. (2021), *Frontiers in Marine Science*** — CNN for humpback song over 187,000+ hours across 13 sites; average precision 0.97, AUC-ROC 0.992. https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2021.607321/full
- **Miller et al. (2023), *Remote Sensing in Ecology and Conservation*** — DL detector outperforms an experienced human observer on blue whale D-calls (double-observer analysis). https://zslpublications.onlinelibrary.wiley.com/doi/full/10.1002/rse2.297
- **Baleen whale social-call CNN (2021), *JASA*** — automatic detection AND classification of baleen social calls with CNNs (directly relevant to our detection+classification dual goal). https://pubs.aip.org/asa/jasa/article/149/5/3635/607542
- Additional species-specific CNNs: sperm whale (Bermant et al. 2019), killer whale (Bergler et al. ORCA-SPOT 2019), fin whale pulses, beluga — establishing CNNs as the de-facto standard across cetaceans.

## 3. Benchmarks for marine bioacoustic detection
- **Schall et al. (2024), *Remote Sensing in Ecology and Conservation*** — "Deep learning in marine bioacoustics: a benchmark for baleen whale detection." Trains three CNN models on open-access datasets; a useful template for how to frame a multi-model benchmark contribution. https://zslpublications.onlinelibrary.wiley.com/doi/abs/10.1002/rse2.392
- **BEANS / BirdSet-style benchmarks** — broader bioacoustic benchmark suites (to verify and cite).

## 4. Bowhead-specific deep learning
- **Thode et al. (2012)** — the original automated bowhead detector/localizer this paper improves upon (hand-crafted features + shallow NN). Detect-only, no classification.
- **CNN-LSTM + adaptive SWT (2023), *Remote Sensing* (MDPI)** — bowhead recognition in the Beaufort Sea; adaptive synchrosqueezing-wavelet-transform features; 10-fold CV mean accuracy 92.85%. Note: focuses on recognition/classification, smaller-scale than our 8.7M-annotation dataset. https://www.mdpi.com/2072-4292/15/22/5346

> **Contribution framing:** our dataset (8.69M manually annotated call detections, 2007–2014 DASAR network) is far larger and more diverse (multiple call types + airgun contamination) than prior bowhead DL studies, and we directly compare *unsupervised representation learning* (autoencoder) against *supervised transfer learning* from foundation models — a comparison not yet reported for bowheads.

## 5. Transfer learning from bioacoustic foundation models
This is the most active and directly relevant frontier for benchmarks (2) and (3).

- **Ghani, Denton et al. (2023), *Scientific Reports*** — "Global birdsong embeddings enable superior transfer learning for bioacoustic classification." Foundational result: BirdNET (and Perch/bird) embeddings transfer across taxa (bats, **marine mammals**, frogs) and beat generic audio models. Justifies using bird-trained embeddings for whales. https://www.nature.com/articles/s41598-023-49989-z
- **Burns et al. (2025), Google DeepMind — "Perch 2.0 transfers 'whale' to underwater tasks"** (arXiv:2512.03219; NeurIPS 2025 AI for Animal Communication workshop). **The single most relevant reference — a near-template for our benchmarks (2) & (3).**
  - Compared 7 embedding models: **Perch 2.0, Perch 1.0, SurfPerch, Google Multispecies Whale Model (GMWM / `multispecies_whale`, Harvey 2024), BirdNET 2.3, AVES-bio, BirdAVES.**
  - Protocol: **few-shot linear probing** with k ∈ {4, 8, 16, 32} recording-level embeddings per class; logistic-regression classifier; 5 independent repeats; **ROC-AUC, one-vs-all.**
  - Models run at **native sample rates (16–48 kHz), 3–5 s windows; NO frequency shifting/band adaptation applied.**
  - Tasks: DCLDE 2026 killer-whale species/ecotype (~200k annotations), NOAA PIPAN baleen whales (30-s weak labels), ReefSet.
  - Headline (ROC-AUC, k=8/k=16): Perch 2.0 won most cetacean tasks (DCLDE species 0.970/0.977; ecotype 0.917/0.945; ReefSet 0.975/0.981) **but BirdNET 2.3 matched/beat it on NOAA baleen whales (0.990/0.991 vs Perch 0.983/0.989).** Takeaway: broadly-trained terrestrial models give high-quality embeddings that dramatically cut labeled-data needs ("agile modeling").
  - https://arxiv.org/abs/2512.03219
- **SurfPerch — Williams et al. (2024)** — "Leveraging tropical reef, bird and unrelated sounds for superior transfer learning in marine bioacoustics" (arXiv:2404.16436). Marine-tuned Perch variant. https://arxiv.org/pdf/2404.16436

### 5a. The low-frequency mismatch problem & the documented fix
Bird foundation models are most sensitive in the mid-frequency (≈1–8 kHz) band of birdsong; bowhead calls sit at **25–500 Hz, sampled at 1 kHz**. Naïve transfer underuses the models. The validated workaround (from the NARW upcall band, ≈50–500 Hz — nearly identical to bowhead simple calls):
- **NARW deep audio embeddings (2025), bioRxiv 2025.07.11.664307** — uses BirdNET embeddings (1024-d) for NARW detection, communication, and individual ID. Preprocessing recipe: resample to 2 kHz, mean-amplitude normalize, bandpass 50–500 Hz, zero-pad to 3 s, **then speed up the signal ~10× to shift 50–500 Hz → 500–5000 Hz into the bird-sensitive band.** Found **speed-up outperforms pitch-shift** (preserves the model's expected time-frequency structure better). https://www.biorxiv.org/content/10.1101/2025.07.11.664307v1.full

## 6. Implications for our benchmark design
1. **Mirror the Perch 2.0 whale protocol** for benchmarks (2)/(3): frozen embeddings + logistic-regression/MLP linear probe, few-shot sweeps (k = 4/8/16/32), 5 repeats, report ROC-AUC + precision-recall. This makes our results directly comparable to a 2025 SOTA paper.
2. **Add the Google Multispecies Whale Model (`multispecies_whale`, Harvey 2024) as a 4th transfer baseline** — it is marine-mammal-specific and was the relevant comparator in the Perch 2.0 whale study; arguably a fairer "transfer" baseline than bird-only models.
3. **Run two preprocessing arms for the bird/Perch models:** (a) native (no shift, as in Burns 2025) and (b) **10× speed-up shift** per the NARW recipe — and report both. This both follows SOTA and tests whether the frequency-shift trick helps on bowheads.
4. **Grouped train/test splits** (by unique-call or date×site) are essential to avoid leakage from multiple DASAR views of the same call — applies to all four pipelines including the existing AE+kNN.
5. **Evaluate at realistic prevalence (~1:8 call:transient)**, not the balanced 1:1 training ratio.

## 7. To verify / still to add
- Full bibliographic details + DOIs for all entries; reconcile with `Bowhead_AI_references.bib`.
- Confirm GMWM citation (Harvey et al. 2024; Allen et al. Bryde's 2024) and Perch 2.0 species count (~14,597 species reported).
- Add ORCA-SPOT (Bergler 2019), Bermant sperm whale (2019), AVES (Hagiwara 2023) primary cites.
- Add autoencoder/representation-learning cites already in the draft (Ozanich 2021, Chien 2023, Guo 2017).
