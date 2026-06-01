"""Transfer-learning benchmarks (BirdNET, Perch 2.0, GMWM).

These pit supervised transfer learning against the unsupervised autoencoder.
Protocol mirrors Burns et al. 2025 ("Perch 2.0 transfers 'whale'"): frozen
embeddings -> linear probe, few-shot sweep k in {4,8,16,32}, 5 repeats, ROC-AUC.

NOTE: BirdNET/Perch consume AUDIO WAVEFORMS, not the dB-SNR spectrogram images
used by the autoencoder pipeline. The transfer benchmarks therefore require the
underlying ~5 s audio clips (omni channel, 1 kHz) for each transient detection.

Heavy backends (tensorflow, tensorflow_hub, librosa) are lazy-imported in
``embedders.py`` so this package imports fine on a machine without them.
"""

from bowhead.transfer.preprocess import FrequencyShiftConfig, preprocess_clip
from bowhead.transfer.probe import LinearProbe, few_shot_eval, ProbeScorer

__all__ = [
    "FrequencyShiftConfig",
    "preprocess_clip",
    "LinearProbe",
    "few_shot_eval",
    "ProbeScorer",
]
