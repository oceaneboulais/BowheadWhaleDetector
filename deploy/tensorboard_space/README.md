---
title: Bowhead Whale Detector — TensorBoard
emoji: 🐋
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Bowhead Whale Detector — training dashboard

Live TensorBoard for the supervised CNN bowhead-call detector
([source repo](https://github.com/oceaneboulais/BowheadWhaleDetector)).

Shows, per run (`scratch` vs `warm-start` arms):

- **SCALARS** — training loss + held-out validation ROC-AUC per epoch, and final
  test metrics at realistic (~1:8 call:transient) prevalence.
- **HPARAMS** — warm-start vs random-init ablation compared side by side.

The CNN mirrors the encoder of a 32-D convolutional autoencoder so the
unsupervised features warm-start the classifier. Companion code for
*"Enhancing bowhead whale detection using convolutional autoencoders"*
(Thode 2026, JASA).

## How the logs get here

TensorBoard event files are written to `runs/<tag>/` during training and copied
into this Space's `logs/` directory. To refresh:

```bash
rm -rf deploy/tensorboard_space/logs && cp -r runs deploy/tensorboard_space/logs
# then push this folder to the Space (see repo README "Portfolio / hosting")
```
