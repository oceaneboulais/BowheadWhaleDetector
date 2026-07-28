"""
Orin AGX Backend: Serve embedding viewer + stream .wav files
"""
from flask import Flask, send_file, jsonify
from pathlib import Path
import json
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== CONFIGURATION: Update these paths for your Orin =====
HTML_FILE = Path("/path/to/BowheadWhaleDetector/docs/eval_embedding_visualization.html")
AUDIO_DIR = Path("/mnt/audio_files")  # Directory containing .wav files
# Example structure: /mnt/audio_files/2024_site3_sample_001.wav
# =========================================================

# Load embedding metadata (for sample lookup)
EMBEDDING_JSON = Path(__file__).parent.parent / "eval" / "embedding_metadata.json"


def get_audio_filename(sample_id):
    """
    Convert sample ID to audio filename.
    Adjust this function based on your naming convention.
    
    Examples:
    - sample_id "001" → "2024_site3_sample_001.wav"
    - sample_id "abc123" → "abc123.wav"
    - sample_id with date "20240715_001" → "20240715_001.wav"
    """
    # Option 1: Direct .wav extension
    return f"{sample_id}.wav"
    
    # Option 2: With date prefix (uncomment if needed)
    # return f"2024_{sample_id}.wav"
    
    # Option 3: With site/folder structure
    # return f"site3/{sample_id}.wav"


@app.route('/')
def index():
    """Serve the interactive HTML embedding viewer"""
    try:
        with open(HTML_FILE, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return jsonify({
            "error": f"HTML file not found at {HTML_FILE}",
            "message": "Generate it first with: python -m bowhead.eval.build_embedding_visualization"
        }), 404


@app.route('/api/audio/<sample_id>')
def get_audio(sample_id):
    """
    Stream audio file for a sample.
    
    URL: /api/audio/sample_001
    Returns: audio/wav stream or 404 if not found
    """
    audio_filename = get_audio_filename(sample_id)
    audio_path = AUDIO_DIR / audio_filename
    
    logger.info(f"Requested audio: {sample_id} → {audio_path}")
    
    if not audio_path.exists():
        logger.warning(f"Audio file not found: {audio_path}")
        return jsonify({
            "error": "Audio file not found",
            "requested_path": str(audio_path),
            "sample_id": sample_id
        }), 404
    
    try:
        return send_file(
            audio_path,
            mimetype='audio/wav',
            as_attachment=False,  # Inline playback
            download_name=f"{sample_id}.wav"
        )
    except Exception as e:
        logger.error(f"Error serving audio: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/metadata')
def get_metadata():
    """Return embedding metadata (samples, projections, etc.)"""
    try:
        if EMBEDDING_JSON.exists():
            with open(EMBEDDING_JSON, 'r') as f:
                return jsonify(json.load(f))
        else:
            return jsonify({"error": "Metadata not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "html_file": str(HTML_FILE),
        "audio_dir": str(AUDIO_DIR),
        "audio_files_available": len(list(AUDIO_DIR.glob("*.wav"))) if AUDIO_DIR.exists() else 0
    })


if __name__ == '__main__':
    # Development server (don't use in production)
    app.run(host='0.0.0.0', port=8000, debug=True)
    
    # For production on Orin AGX, run with:
    # gunicorn --workers 4 --bind 0.0.0.0:8000 --timeout 300 app:app
