# Quick Start Guide: Bowhead Viewer on Orin AGX

## Prerequisites
- NVIDIA Orin AGX with JetPack OS
- `.wav` audio files organized with matching filenames
- Flask backend (see `app.py`)

## Quick Setup (5 minutes)

### 1. SSH into Orin
```bash
ssh orin@<orin-ip>
```

### 2. Download and run setup script
```bash
cd /tmp
curl -O https://raw.githubusercontent.com/oceaneboulais/BowheadWhaleDetector/add-results-dashboards/bowhead/serve/setup-orin.sh
chmod +x setup-orin.sh
./setup-orin.sh
```

### 3. Configure app.py with your paths
```bash
nano /opt/bowhead-viewer/BowheadWhaleDetector/bowhead/serve/app.py
```

**Update these lines:**
```python
HTML_FILE = Path("/opt/bowhead-viewer/BowheadWhaleDetector/docs/eval_embedding_visualization.html")
AUDIO_DIR = Path("/mnt/storage/audio_files")  # <- Your .wav files location
```

### 4. Test locally
```bash
cd /opt/bowhead-viewer/BowheadWhaleDetector
gunicorn --workers 4 --bind 0.0.0.0:8000 --timeout 300 bowhead.serve.app:app
```

Visit: `http://localhost:8000/`

### 5. Set up public access (Cloudflare Tunnel)

#### Option A: Cloudflare Tunnel (Recommended - works with any internet)
```bash
cloudflared tunnel login
cloudflared tunnel create bowhead-viewer
cloudflared tunnel route dns bowhead-viewer yourdomain.com

# Create config file
mkdir ~/.cloudflared
cat > ~/.cloudflared/config.yml <<EOF
tunnel: bowhead-viewer
credentials-file: ~/.cloudflared/<UUID>.json

ingress:
  - hostname: bowhead-viewer.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
EOF

# Test tunnel
cloudflared tunnel run bowhead-viewer
```

**Public URL:** `https://bowhead-viewer.yourdomain.com/`

#### Option B: Local Network (Tailscale)
```bash
# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# Authenticate
sudo tailscale up

# Your local network URL: https://<your-orin>.ts.net:8000/
```

### 6. Install as systemd service (auto-start)
```bash
sudo cp /opt/bowhead-viewer/BowheadWhaleDetector/bowhead/serve/bowhead-viewer.service \
        /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable bowhead-viewer
sudo systemctl start bowhead-viewer

# Check status
sudo systemctl status bowhead-viewer

# View logs
journalctl -u bowhead-viewer -f
```

---

## Audio File Structure

The server expects audio files in a specific directory with filenames matching sample IDs.

### Example Structure:
```
/mnt/audio_files/
├── sample_001.wav
├── sample_002.wav
├── sample_003.wav
└── ...
```

### Custom Naming Convention?
If your files use a different naming pattern, edit the `get_audio_filename()` function in `app.py`:

```python
def get_audio_filename(sample_id):
    # If files are: "bowhead_20240715_001.wav"
    return f"bowhead_{sample_id}.wav"
```

---

## Testing

### 1. Health check
```bash
curl http://localhost:8000/health
```

### 2. Test audio streaming
```bash
curl http://localhost:8000/api/audio/sample_001 > test.wav
```

### 3. View logs
```bash
journalctl -u bowhead-viewer -f
```

---

## Troubleshooting

### Audio files not found
- Check path in `AUDIO_DIR` 
- Verify filename matches `get_audio_filename()` function
- Check server logs: `journalctl -u bowhead-viewer -f`

### Cloudflare Tunnel not working
```bash
# Check tunnel status
cloudflared tunnel list
cloudflared tunnel info bowhead-viewer

# Re-authenticate
cloudflared tunnel login
```

### Port already in use
```bash
# Check what's on port 8000
lsof -i :8000

# Use different port in systemd service or gunicorn
```

---

## Performance Tips

- **Workers**: Set to number of Orin CPU cores (typically 8-12)
- **Timeout**: 300s is good for large files; adjust for your network
- **Cache**: Add reverse proxy cache (nginx) in front of Flask if many requests

---

## Monitoring

```bash
# Systemd service status
systemctl status bowhead-viewer

# Real-time logs
journalctl -u bowhead-viewer -f

# Server health
curl http://localhost:8000/health | python -m json.tool

# Check available audio files
ls /mnt/audio_files/*.wav | wc -l
```
