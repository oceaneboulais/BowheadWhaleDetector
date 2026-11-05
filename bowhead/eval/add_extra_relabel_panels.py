"""Add "before vs. after relabeling" panels for the Moan Detector and
BirdNET/Perch 2.0 baselines to docs/benchmark_pr_curves.html.

Same idea as ``bowhead.eval.add_relabel_panel`` (scratch CNN): each detector
is scored ONCE on the eval set (Moan Detector: full 199,825 samples; BirdNET:
now also the full 199,825-image eval set, protocol-matched to the other
CNN-family rows), then its precision-recall / FDR-vs-miss curves are
computed twice, once per label set, isolating the effect of dataset
relabeling from the detector itself.

Reads:
    runs/moan_detector_relabel_100k_matched.json         (bowhead.eval.run_moan_relabel_curves)
    runs/birdnet_relabel_100k_matched_full199825.json    (bowhead.eval.run_birdnet_relabel_curves_full)

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.eval.add_extra_relabel_panels
"""

from __future__ import annotations

import json
from pathlib import Path

from bowhead.eval.relabel_panel_common import section_html

_REPO_ROOT = Path(__file__).resolve().parents[2]
MOAN_JSON = _REPO_ROOT / "runs" / "moan_detector_relabel_100k_matched.json"
BIRDNET_JSON = _REPO_ROOT / "runs" / "birdnet_relabel_100k_matched_full199825.json"
HTML_PATH = _REPO_ROOT / "docs" / "benchmark_pr_curves.html"
MARKER = '<div id="extra-relabel-panels-marker" style="display:none"></div>'


def main() -> None:
    moan = json.loads(MOAN_JSON.read_text())
    birdnet = json.loads(BIRDNET_JSON.read_text())

    section = MARKER
    section += section_html(
        "Relabeling Ablation — Moan Detector, Before vs. After Review",
        "Moan Detector (classical multi-band energy detector) — Before vs. After Relabeling",
        "the classical multi-band energy detector (no training; a faithful port of the "
        "Baumgartner &amp; Mussoline 2011-style detector used in "
        "<code style='font-size:11px;background:#efefef;padding:1px 5px;border-radius:3px'>"
        "matlab/matlab/MultipleBandEnergyDetector.m</code>)",
        moan,
    )
    section += section_html(
        "Relabeling Ablation — BirdNET / Perch 2.0, Before vs. After Review",
        "BirdNET / Perch 2.0 (Griffin-Lim reconstructed audio) — Before vs. After Relabeling",
        "the frozen BirdNET/Perch 2.0 linear probe on Griffin-Lim reconstructed audio "
        f"(probe fit on a {birdnet['_meta']['n_train']:,}-image train draw, evaluated on the "
        f"full {birdnet['_meta']['n']:,}-image eval set, protocol-matched to the other "
        "CNN-family rows)",
        birdnet,
        caveat_html=birdnet["_meta"]["caveat"],
    )

    html = HTML_PATH.read_text()
    marker_idx = html.find(MARKER)
    if marker_idx != -1:
        html = html[:marker_idx]
    else:
        body_idx = html.find("</body>")
        html = html[:body_idx] if body_idx != -1 else html
    html = html.rstrip() + "\n" + section + "\n</body>\n</html>\n"
    HTML_PATH.write_text(html)
    print(f"Inserted moan_detector + birdnet relabeling panels into {HTML_PATH}")


if __name__ == "__main__":
    main()
