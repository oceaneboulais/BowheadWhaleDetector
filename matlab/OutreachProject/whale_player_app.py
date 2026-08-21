#!/usr/bin/env python3
"""
Bowhead Whale Call Interactive Player
Browser-based Theremin-style synthesizer for whale vocalizations

This Flask application serves spectrograms from the whale call database
and enables interactive playback through a web-based synthesizer.
"""

import os
import glob
import json
import numpy as np
from flask import Flask, render_template, jsonify, send_file
from scipy.io import loadmat
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

app = Flask(__name__)

# Dataset configuration
DATA_DIR = "/Users/oboulais/Public/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir"

# Call type metadata based on literature
CALL_TYPE_INFO = {
    1: {"name": "Simple Upsweep", "color": "#3498db", "description": "Rising frequency sweep"},
    2: {"name": "Simple Downsweep", "color": "#e74c3c", "description": "Falling frequency sweep"},
    3: {"name": "Constant Tone", "color": "#2ecc71", "description": "Relatively constant frequency"},
    4: {"name": "Complex Call", "color": "#f39c12", "description": "Modulated multi-component"},
    5: {"name": "Pulse Train", "color": "#9b59b6", "description": "Repetitive pulse"},
    6: {"name": "Harmonic Call", "color": "#1abc9c", "description": "Harmonic structure"},
    7: {"name": "Variable Call", "color": "#e67e22", "description": "Variable frequency pattern"}
}

# Cache for file listings
_file_cache = {}


def get_call_files_by_type(call_type: int, max_per_type: int = 50) -> List[str]:
    """Get list of files for a specific call type."""
    cache_key = f"type_{call_type}"
    
    if cache_key not in _file_cache:
        pattern = os.path.join(DATA_DIR, f"*Type{call_type}.mat")
        files = sorted(glob.glob(pattern))[:max_per_type]
        _file_cache[cache_key] = files
    
    return _file_cache[cache_key]


def load_spectrogram(filepath: str) -> np.ndarray:
    """Load SNR_gram from .mat file."""
    try:
        data = loadmat(filepath)
        return data.get('SNR_gram', None)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def extract_frequency_contour(spectrogram: np.ndarray, 
                              freq_min: float = 50, 
                              freq_max: float = 500,
                              num_freq_bins: int = 121) -> List[float]:
    """
    Extract dominant frequency contour from spectrogram.
    Returns list of frequencies (in Hz) over time.
    
    Args:
        spectrogram: 2D array (freq x time)
        freq_min: Minimum frequency in Hz
        freq_max: Maximum frequency in Hz
        num_freq_bins: Number of frequency bins
    """
    if spectrogram is None:
        return []
    
    # Frequency bins (assuming linear spacing)
    freq_bins = np.linspace(freq_min, freq_max, num_freq_bins)
    
    # Find peak frequency for each time step
    contour = []
    for time_idx in range(spectrogram.shape[1]):
        if spectrogram.shape[0] == len(freq_bins):
            spectrum = spectrogram[:, time_idx]
        else:
            # If dimensions don't match, interpolate
            spectrum = np.interp(
                np.linspace(0, 1, num_freq_bins),
                np.linspace(0, 1, spectrogram.shape[0]),
                spectrogram[:, time_idx]
            )
        
        # Find peak frequency (weighted average for smoother results)
        if np.max(spectrum) > 0:
            # Weighted centroid
            peak_freq = np.sum(freq_bins * spectrum) / np.sum(spectrum)
        else:
            peak_freq = freq_min
        
        contour.append(float(peak_freq))
    
    return contour


def create_spectrogram_image(spectrogram: np.ndarray) -> str:
    """Create base64-encoded spectrogram image."""
    if spectrogram is None:
        return ""
    
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='black')
    
    # Plot spectrogram
    im = ax.imshow(spectrogram, aspect='auto', origin='lower', 
                   cmap='viridis', interpolation='bilinear')
    ax.set_xlabel('Time', color='white')
    ax.set_ylabel('Frequency Bin', color='white')
    ax.tick_params(colors='white')
    
    # Remove white borders
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', facecolor='black', edgecolor='none', dpi=100)
    plt.close(fig)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{image_base64}"


@app.route('/')
def index():
    """Serve the main interactive player interface."""
    return render_template('whale_player.html', call_types=CALL_TYPE_INFO)


@app.route('/api/call_types')
def get_call_types():
    """Return available call types with metadata and counts."""
    result = {}
    for call_type in range(1, 8):
        files = get_call_files_by_type(call_type, max_per_type=100)
        result[call_type] = {
            **CALL_TYPE_INFO[call_type],
            "count": len(files),
            "sample_files": [os.path.basename(f) for f in files[:5]]
        }
    return jsonify(result)


@app.route('/api/call/<int:call_type>/<int:index>')
def get_call_data(call_type: int, index: int):
    """
    Get spectrogram and frequency contour for a specific call.
    
    Args:
        call_type: Type number (1-7)
        index: Index within that type
    """
    if call_type < 1 or call_type > 7:
        return jsonify({"error": "Invalid call type"}), 400
    
    files = get_call_files_by_type(call_type, max_per_type=100)
    
    if index < 0 or index >= len(files):
        return jsonify({"error": "Invalid index"}), 400
    
    filepath = files[index]
    spectrogram = load_spectrogram(filepath)
    
    if spectrogram is None:
        return jsonify({"error": "Failed to load spectrogram"}), 500
    
    # Extract frequency contour
    frequency_contour = extract_frequency_contour(spectrogram)
    
    # Create spectrogram image
    spec_image = create_spectrogram_image(spectrogram)
    
    return jsonify({
        "filename": os.path.basename(filepath),
        "call_type": call_type,
        "call_name": CALL_TYPE_INFO[call_type]["name"],
        "frequency_contour": frequency_contour,
        "duration_ms": len(frequency_contour) * 10,  # Assuming ~10ms per time step
        "spectrogram_image": spec_image,
        "shape": spectrogram.shape
    })


@app.route('/api/random_call/<int:call_type>')
def get_random_call(call_type: int):
    """Get a random call from a specific type."""
    if call_type < 1 or call_type > 7:
        return jsonify({"error": "Invalid call type"}), 400
    
    files = get_call_files_by_type(call_type, max_per_type=100)
    
    if not files:
        return jsonify({"error": "No files found"}), 404
    
    # Pick random index
    import random
    index = random.randint(0, len(files) - 1)
    
    return get_call_data(call_type, index)


if __name__ == '__main__':
    print("=" * 60)
    print("Bowhead Whale Call Interactive Player")
    print("=" * 60)
    print("\nLoading call database...")
    
    # Pre-cache file listings
    for call_type in range(1, 8):
        files = get_call_files_by_type(call_type, max_per_type=100)
        print(f"  Type {call_type} ({CALL_TYPE_INFO[call_type]['name']}): {len(files)} samples loaded")
    
    print("\n" + "=" * 60)
    print("Starting server...")
    print("Open your browser to: http://localhost:5010")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server")
    
    app.run(debug=True, host='127.0.0.1', port=5010, use_reloader=False)
