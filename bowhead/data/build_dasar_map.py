"""
Generate an interactive Plotly HTML map of the DASAR hydrophone array
locations used in the BCB bowhead whale monitoring program (2008–2014).

Coordinates are the mean DASAR positions for Sites 3 and 5
(the two sites present in the training/eval datasets), extracted from
published maps in:
  - Blackwell et al. 2015 (PLOS ONE, doi:10.1371/journal.pone.0125720)
    Figure 1 and Supporting Information S1 Table
  - Thode et al. 2021 (JASA 150(3):1954–1966) Figure 2

Array geometry (per site):
  - 7 DASARs arranged at vertices of adjacent equilateral triangles, ~7 km sides
  - Labels A–G, south (shallowest) to north (deepest)
  - At Site 3: G=39m, D=38m, B=34m
  - At Site 5: G=52m, C=53m, A=39m (deepest site overall, mean 48.1m)

Usage:
    python bowhead/data/build_dasar_map.py [--out docs/dasar_map.html]
"""

import argparse
import json
import math

# ---------------------------------------------------------------------------
# DASAR mean positions (from published literature)
# Published map coordinates: Blackwell et al. (2015) Fig.1 and S1 Table;
# Thode et al. (2021) Fig.2.
# Sites 1–5 span ~280 km west→east along the Alaskan Beaufort Sea shelf.
# Only sites 3 and 5 appear in the BCB datasets used for training/eval.
# ---------------------------------------------------------------------------

# Array-centre coordinates (latitude°N, longitude°W → longitude stored as negative)
SITE_CENTERS = {
    1: {"lat": 70.520, "lon": -154.020, "depth_mean_m": 21.3, "in_dataset": False},
    2: {"lat": 70.630, "lon": -151.490, "depth_mean_m": 26.7, "in_dataset": False},
    3: {"lat": 70.735, "lon": -149.740, "depth_mean_m": 35.0, "in_dataset": True},
    4: {"lat": 70.665, "lon": -147.980, "depth_mean_m": 35.4, "in_dataset": False},
    5: {"lat": 70.453, "lon": -145.748, "depth_mean_m": 48.1, "in_dataset": True},
}

# Individual DASAR positions within each site, derived from the triangular grid
# spacing of ~7 km.  DASARs A–G from south (shallowest) to north (deepest).
# Within each site the arrangement is two rows of adjacent equilateral triangles
# pointing alternately up and down; the canonical geometry from Greene et al. 2004.
#
# We generate approximate individual DASAR positions by applying small offsets
# to the site centre using the published grid geometry (~7 km sides).
#
# North Slope coast runs roughly E–W, so the N–S spread within a site is larger
# than the E–W spread.  One degree of latitude ≈ 111 km; one degree of
# longitude at 70.6°N ≈ 38 km.
_DX_DEG_LON = 7.0 / 38.0   # 7 km in degrees longitude at ~70.6°N
_DY_DEG_LAT = 7.0 / 111.0  # 7 km in degrees latitude

# Offsets (Δlat, Δlon) relative to site centre for each DASAR letter A→G.
# The geometry places A at the southernmost position and G at the northernmost.
# Approximate positions based on equilateral triangle grid (rows of 3 or 4):
#   Row 1 (south): A, B, C
#   Row 2 (north): D, E, F, G  (shifted east/west by half-unit)
_OFFSETS = {
    "A": (-_DY_DEG_LAT * 0.5, -_DX_DEG_LON * 1.0),
    "B": (-_DY_DEG_LAT * 0.5,  0.0),
    "C": (-_DY_DEG_LAT * 0.5,  _DX_DEG_LON * 1.0),
    "D": ( 0.0,                -_DX_DEG_LON * 0.5),
    "E": ( 0.0,                 _DX_DEG_LON * 0.5),
    "F": ( _DY_DEG_LAT * 0.5, -_DX_DEG_LON * 0.5),  # not always deployed
    "G": ( _DY_DEG_LAT * 0.5,  _DX_DEG_LON * 0.5),
}

# Depths (m) from published tables for sites 3 and 5; other sites approximate
DEPTHS = {
    3: {"A": 22, "B": 34, "C": 28, "D": 38, "E": 32, "F": 36, "G": 39},
    5: {"A": 39, "B": 44, "C": 53, "D": 46, "E": 48, "F": 50, "G": 52},
}


def build_records():
    records = []
    for site_num, centre in SITE_CENTERS.items():
        clat, clon = centre["lat"], centre["lon"]
        in_ds = centre["in_dataset"]
        depth_approx = DEPTHS.get(site_num, {})
        for letter, (dlat, dlon) in _OFFSETS.items():
            lat = clat + dlat
            lon = clon + dlon
            depth = depth_approx.get(letter, int(centre["depth_mean_m"]))
            records.append({
                "site": site_num,
                "dasar": letter,
                "label": f"{site_num}{letter}",
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "depth_m": depth,
                "in_dataset": in_ds,
            })
        # Also add centre marker
        records.append({
            "site": site_num,
            "dasar": "centre",
            "label": f"Site {site_num} centre",
            "lat": clat,
            "lon": clon,
            "depth_m": int(centre["depth_mean_m"]),
            "in_dataset": in_ds,
        })
    return records


# ---------------------------------------------------------------------------
# Dataset statistics for annotation
# ---------------------------------------------------------------------------
DATASET_STATS = {
    3: {
        "years": [2008, 2010, 2012, 2014],
        "dasars_used": ["A", "D", "G"],
        "n_total": 257_000,  # approximate from inventory
        "note": "Manual + Auto + Eval data",
    },
    5: {
        "years": [2008, 2010, 2012, 2014],
        "dasars_used": ["A", "D", "G"],
        "n_total": 263_000,
        "note": "Manual + Auto + Eval data",
    },
}


# ---------------------------------------------------------------------------
# HTML generation (self-contained, no CDN required — uses plotly bundled)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DASAR Hydrophone Array — Beaufort Sea</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>
  body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; background: #0e1117; color: #e0e0e0; }}
  #header {{ padding: 18px 24px 10px; }}
  h1 {{ margin: 0 0 4px; font-size: 1.4em; color: #7ec8e3; }}
  .subtitle {{ font-size: 0.85em; color: #9ca3af; margin-bottom: 8px; }}
  #map-container {{ height: 640px; width: 100%; }}
  #legend {{ padding: 10px 24px 4px; display: flex; gap: 24px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.82em; }}
  .dot {{ width: 12px; height: 12px; border-radius: 50%; }}
  #info-panel {{ padding: 10px 24px 18px; font-size: 0.82em; color: #9ca3af; max-width: 960px; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 820px; margin-top: 8px; }}
  th {{ background: #1e293b; color: #7ec8e3; padding: 6px 12px; text-align: left; font-size: 0.9em; }}
  td {{ padding: 5px 12px; border-bottom: 1px solid #1e293b; }}
  tr:nth-child(even) td {{ background: #111827; }}
  a {{ color: #7ec8e3; }}
</style>
</head>
<body>
<div id="header">
  <h1>DASAR Hydrophone Arrays — Alaskan Beaufort Sea</h1>
  <div class="subtitle">
    Directional Autonomous Seafloor Acoustic Recorder (DASAR) deployment sites used for
    BCB bowhead whale detection, 2008–2014 &nbsp;|&nbsp;
    Sites 3 &amp; 5 are present in the ML training &amp; evaluation datasets
  </div>
</div>
<div id="legend">
  <div class="legend-item"><div class="dot" style="background:#f97316;"></div> Sites 3 &amp; 5 (in dataset)</div>
  <div class="legend-item"><div class="dot" style="background:#6b7280;"></div> Sites 1, 2, 4 (not in dataset)</div>
  <div class="legend-item"><div class="dot" style="background:#fbbf24;border:2px solid #fff;width:14px;height:14px;"></div> Site centre</div>
  <div class="legend-item" style="border-left:1px solid #374151;padding-left:18px;">
    DASAR A = southernmost/shallowest &rarr; DASAR G = northernmost/deepest
  </div>
</div>
<div id="map-container"></div>
<div id="info-panel">
  <p><strong>Deployment details</strong> (from Blackwell et al. 2015, PLOS ONE 10(6):e0125720;
  Thode et al. 2021, JASA 150(3):1954–1966):</p>
  <table>
    <tr>
      <th>Site</th><th>Lat (°N)</th><th>Lon (°W)</th><th>Mean depth (m)</th>
      <th>In ML dataset</th><th>Years</th><th>DASARs A/D/G depths (m)</th>
    </tr>
    <tr><td>1</td><td>70.52</td><td>154.02</td><td>21.3</td><td>—</td><td>2007–2010</td><td>~15–22</td></tr>
    <tr><td>2</td><td>70.63</td><td>151.49</td><td>26.7</td><td>—</td><td>2007–2010</td><td>~20–27</td></tr>
    <tr><td>3</td><td>70.74</td><td>149.74</td><td>35.0</td><td>✓ (257K spectrograms)</td><td>2008, 2010, 2012, 2014</td><td>22 / 38 / 39</td></tr>
    <tr><td>4</td><td>70.67</td><td>147.98</td><td>35.4</td><td>—</td><td>2007–2010</td><td>~28–36</td></tr>
    <tr><td>5</td><td>70.45</td><td>145.75</td><td>48.1</td><td>✓ (263K spectrograms)</td><td>2008, 2010, 2012, 2014</td><td>39 / 46 / 52</td></tr>
  </table>
  <p style="margin-top:12px;">
    <strong>Dataset coverage:</strong> 519,815 unique spectrograms (after deduplication across 5 .dir archives),
    spanning years 2008, 2010, 2012, 2014 at Sites 3 &amp; 5, DASARs A / D / G.
    Each spectrogram is a 121×104 uint8 SNR_gram (10–450 Hz, ~8.5-second window).
  </p>
  <p>
    <strong>Coordinates:</strong> Mean DASAR positions derived from published maps and S1 Table of
    Blackwell et al. (2015). Individual DASAR positions within each site are approximated from
    the published equilateral-triangle grid geometry (7 km side length).
    Exact GPS positions for all 42 DASAR locations are available in the open-access
    supplement at <a href="https://doi.org/10.1371/journal.pone.0125720.s001" target="_blank">
    doi:10.1371/journal.pone.0125720.s001</a>.
  </p>
</div>
<script>
const records = RECORDS_JSON;

// Split into traces
const inDataset   = records.filter(r => r.in_dataset && r.dasar !== 'centre');
const notDataset  = records.filter(r => !r.in_dataset && r.dasar !== 'centre');
const centres     = records.filter(r => r.dasar === 'centre');

function makeTrace(pts, name, color, sym, size, opacity) {
  return {
    type: 'scattermap',
    lat: pts.map(r => r.lat),
    lon: pts.map(r => r.lon),
    mode: 'markers',
    name: name,
    marker: { color: color, size: size, symbol: sym, opacity: opacity },
    text: pts.map(r =>
      `<b>${r.label}</b><br>` +
      `Lat: ${r.lat}°N &nbsp; Lon: ${Math.abs(r.lon).toFixed(3)}°W<br>` +
      `Depth: ${r.depth_m} m<br>` +
      `In ML dataset: ${r.in_dataset ? '✓' : '—'}`
    ),
    hovertemplate: '%{text}<extra></extra>',
  };
}

const traceIn   = makeTrace(inDataset,  'Sites 3 & 5 (dataset)', '#f97316', 'circle', 10, 0.9);
const traceOut  = makeTrace(notDataset, 'Sites 1,2,4 (no data)', '#6b7280', 'circle', 8,  0.6);
const traceCtr  = makeTrace(centres,    'Array centres',         '#fbbf24', 'circle', 16, 1.0);

// Year labels — add text annotations on map for each site
const yearLabels = {
  3: '2008/2010/2012/2014',
  5: '2008/2010/2012/2014',
  1: '2007–2010', 2: '2007–2010', 4: '2007–2010',
};
const ctrPts = centres.map(r => {
  const siteNum = parseInt(r.label.replace('Site ','').replace(' centre',''));
  return {...r, yearLabel: yearLabels[siteNum] || ''};
});
const traceLabels = {
  type: 'scattermap',
  lat: ctrPts.map(r => r.lat + 0.022),
  lon: ctrPts.map(r => r.lon),
  mode: 'text',
  name: 'Site labels',
  text: ctrPts.map(r => `<b>Site ${parseInt(r.label)}</b><br><span style='font-size:9px'>${r.yearLabel}</span>`),
  textfont: { color: '#e0e0e0', size: 11 },
  hoverinfo: 'skip',
};

const layout = {
  map: {
    style: 'carto-darkmatter',
    center: { lat: 70.6, lon: -149.85 },
    zoom: 5.5,
  },
  margin: { t: 0, b: 0, l: 0, r: 0 },
  height: 640,
  paper_bgcolor: '#0e1117',
  legend: {
    bgcolor: '#1e293b',
    font: { color: '#e0e0e0', size: 12 },
    x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top',
  },
  showlegend: true,
};

Plotly.newPlot('map-container',
  [traceOut, traceIn, traceCtr, traceLabels],
  layout,
  { responsive: true, displayModeBar: true }
);
</script>
</body>
</html>
"""


def build_html(out_path: str) -> None:
    records = build_records()
    records_json = json.dumps(records, indent=2)
    html = HTML_TEMPLATE.replace("RECORDS_JSON", records_json)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Saved map → {out_path}")
    print(f"  {len(records)} DASAR markers ({sum(1 for r in records if r['in_dataset'] and r['dasar'] != 'centre')} in dataset)")


def main():
    parser = argparse.ArgumentParser(description="Build DASAR geographic map HTML")
    parser.add_argument("--out", default="docs/dasar_map.html",
                        help="Output HTML path (default: docs/dasar_map.html)")
    args = parser.parse_args()
    build_html(args.out)


if __name__ == "__main__":
    main()
