"""Add "before vs. after relabeling" panels for the Moan Detector and
BirdNET/Perch 2.0 baselines to docs/benchmark_pr_curves.html.

Same idea as ``bowhead.eval.add_relabel_panel`` (scratch CNN): each detector
is scored ONCE on the eval set (Moan Detector: full 199,825 samples; BirdNET:
a subsample, since Griffin-Lim reconstruction is CPU-bound), then its
precision-recall / FDR-vs-miss curves are computed twice, once per label set,
isolating the effect of dataset relabeling from the detector itself.

Reads:
    runs/moan_detector_relabel_100k_matched.json  (bowhead.eval.run_moan_relabel_curves)
    runs/birdnet_relabel_100k_matched.json        (bowhead.eval.run_birdnet_relabel_curves)

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.eval.add_extra_relabel_panels
"""

from __future__ import annotations

import json
from pathlib import Path

from bowhead.eval.relabel_panel_common import section_html

_REPO_ROOT = Path(__file__).resolve().parents[2]
MOAN_JSON = _REPO_ROOT / "runs" / "moan_detector_relabel_100k_matched.json"
BIRDNET_JSON = _REPO_ROOT / "runs" / "birdnet_relabel_100k_matched.json"
HTML_PATH = _REPO_ROOT / "docs" / "benchmark_pr_curves.html"


def main() -> None:
    moan = json.loads(MOAN_JSON.read_text())
    birdnet = json.loads(BIRDNET_JSON.read_text())

    section = section_html(
        "Relabeling Ablation — Moan Detector, Before vs. After Review",
        "Moan Detector (classical multi-band energy detector) — Before vs. After Relabeling",
        "the classical multi-band energy detector (no training; a faithful port of the "
        "Baumgartner &amp; Mussoline 2011-style detector used in "
        "<code style='font-size:11px;background:#efefef;padding:1px 5px;border-radius:3px'>"
        "matlab/matlab/MultipleBandEnergyDetector.m</code>)",
        moan,
    )
    section += section_html(
        "Relabeling Ablation — BirdNET / Perch 2.0 (Subsample), Before vs. After Review",
        "BirdNET / Perch 2.0 (Griffin-Lim reconstructed audio) — Before vs. After Relabeling",
        "the frozen BirdNET/Perch 2.0 linear probe on Griffin-Lim reconstructed audio "
        f"(probe fit on a {birdnet['_meta']['n_train']:,}-image train subsample, evaluated on a "
        f"{birdnet['_meta']['n']:,}-image subsample of the eval set)",
        birdnet,
        caveat_html=birdnet["_meta"]["caveat"],
    )

    html = HTML_PATH.read_text()
    html = html.replace("</body>", section + "\n</body>")
    HTML_PATH.write_text(html)
    print(f"Inserted moan_detector + birdnet relabeling panels into {HTML_PATH}")


if __name__ == "__main__":
    main()
