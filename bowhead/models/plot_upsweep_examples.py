"""Re-render the exact 18 upsweep examples shown in Upsweep_examples.jpg in a
larger, higher-contrast, more legible 3x6 grid.

The original jpg (produced by matlab/matlab/master_plot_call_type_samples.m)
uses a MATLAB default colormap, tiny per-panel titles with raw .mat stems,
and axis labels/ticks only on the outer row/column, all at modest
resolution. This script locates the same 18 raw .mat spectrograms (by stem,
read off the original jpg's titles) and re-plots them with a larger figure,
the viridis colormap used elsewhere in the paper, consistent Time (s) /
Frequency (Hz) axis labels and extents, and letter-labeled panels instead of
raw filenames.

Run:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.models.plot_upsweep_examples
"""

from __future__ import annotations

import argparse
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

# Exact stems shown (title order, row-major) in the original Upsweep_examples.jpg.
UPSWEEP_STEMS = [
    "S314G0T20140927T123134_Type1", "S314G0T20140828T000748_Type1",
    "S314G0T20140901T195520_Type1", "S314D0T20140901T051412_Type1",
    "S514A0T20140917T203403_Type1", "S310D0T20100913T035102_Type1",
    "S314G0T20140818T051922_Type1", "S314A0T20140927T073704_Type1",
    "S314A0T20140917T181340_Type1", "S514A0T20140917T234919_Type1",
    "S514D0T20140917T235316_Type1", "S514G0T20140917T114643_Type1",
    "S314G0T20140901T210929_Type1", "S314D0T20140828T034923_Type1",
    "S514A0T20140917T154213_Type1", "S514D0T20140917T150739_Type1",
    "S514A0T20140917T123512_Type1", "S514A0T20140917T095055_Type1",
]

RAW_DATASET_DIR = (
    "/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/"
    "Unsupervised_database_MostlyManual.dir"
)

N_ROWS, N_COLS = 3, 6


def _load_snr_gram(stem: str, raw_dir: str) -> np.ndarray:
    path = Path(raw_dir) / f"{stem}.mat"
    if not path.exists():
        raise FileNotFoundError(f"Could not locate {stem}.mat under {raw_dir}")
    mat = sio.loadmat(str(path), simplify_cells=True)
    return np.asarray(mat["SNR_gram"], dtype=np.uint8)


def _render(stems: list[str], raw_dir: str, out_path: Path) -> None:
    images = [_load_snr_gram(stem, raw_dir) for stem in stems]

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(3.0 * N_COLS, 3.1 * N_ROWS))
    duration_s, freq_hz = 3.0, 500.0
    extent = (0.0, duration_s, 0.0, freq_hz)
    letters = string.ascii_lowercase
    for idx, (img, ax) in enumerate(zip(images, axes.flat)):
        row, col = divmod(idx, N_COLS)
        ax.imshow(img, cmap="viridis", origin="lower", aspect="auto", extent=extent)
        ax.set_title(f"({letters[idx]})", fontsize=13, fontweight="bold")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([0, 100, 200, 300, 400, 500])
        ax.tick_params(labelsize=10)
        if row == N_ROWS - 1:
            ax.set_xlabel("Time (s)", fontsize=12)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("Frequency (Hz)", fontsize=12)
        else:
            ax.set_yticklabels([])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=RAW_DATASET_DIR)
    parser.add_argument("--out", default="paper/Figures/Upsweep_examples_reformatted.png")
    args = parser.parse_args()

    _render(UPSWEEP_STEMS, args.raw_dir, Path(args.out))


if __name__ == "__main__":
    main()
