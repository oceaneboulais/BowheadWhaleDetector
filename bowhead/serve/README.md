# Complete Orin AGX Architecture - Bowhead Viewer

## System Diagram

```
┌─────────────────────────────────────────────────────────┐
│          Browser (macOS/Windows/Linux/Mobile)          │
│     https://bowhead-viewer.yourdomain.com              │
└──────────────────────┬────────────────────────────────┘
                       │
                       │ HTTPS (via Cloudflare Tunnel)
                       │
┌──────────────────────▼────────────────────────────────┐
│     NVIDIA Orin AGX (at your lab/home)                 │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │ Flask + Gunicorn (bowhead.serve.app)           │  │
│  │                                                  │  │
│  │ GET /                  → Serve HTML viewer      │  │
│  │ GET /api/audio/{id}    → Stream .wav file      │  │
│  │ GET /health            → Status check           │  │
│  └─────────────────────────────────────────────────┘  │
│                        ▲                               │
│                        │                               │
│  ┌─────────────────────┴─────────────────────────┐  │
│  │ Local Filesystem                              │  │
│  │                                               │  │
│  │ /opt/bowhead-viewer/BowheadWhaleDetector/   │  │
│  │   ├─ docs/                                   │  │
│  │   │  └─ eval_embedding_visualization.html  │  │
│  │   └─ bowhead/serve/app.py                   │  │
│  │                                               │  │
│  │ /mnt/audio_files/  (your .wav collection)    │  │
│  │   ├─ sample_001.wav                         │  │
│  │   ├─ sample_002.wav                         │  │
│  │   └─ ...                                     │  │
│  └─────────────────────────────────────────────┘  │
│                                                     │
│  Systemd Service: bowhead-viewer (auto-restart)    │
│  Cloudflare Tunnel: Secure public access          │
└─────────────────────────────────────────────────────┘
```

## File Structure

```
BowheadWhaleDetector/
├── bowhead/
│   ├── serve/                          ← NEW: Server module
│   │   ├── __init__.py
│   │   ├── app.py                      ← Flask backend (main)
│   │   ├── requirements.txt            ← Dependencies
│   │   ├── setup-orin.sh              ← Automated setup
│   │   ├── bowhead-viewer.service     ← Systemd config
│   │   ├── DEPLOYMENT.md              ← Full guide
│   │   └── AUDIO_INTEGRATION.md       ← Audio API docs
│   ├── eval/
│   │   └── build_embedding_visualization.py  ← Generate HTML viewer
│   └── ...
│
└── docs/
    └── eval_embedding_visualization.html   ← Interactive viewer (generated)
```

## Deployment Workflow

### Step 1: Generate HTML Viewer (on your Mac)
```bash
python -m bowhead.eval.build_embedding_visualization \
  --data data/spectrograms_eval_v2.npz \
  --out docs/eval_embedding_visualization.html
```

### Step 2: Copy to Orin AGX or push to repo
```bash
# Option A: Push to GitHub
git add bowhead/serve/
git add docs/eval_embedding_visualization.html
git commit -m "Add server backend and embedding viewer"
git push origin add-results-dashboards

# Then on Orin:
cd /opt/bowhead-viewer/BowheadWhaleDetector
git pull
```

### Step 3: Configure Orin Paths
```bash
ssh orin@<orin-ip>
nano /opt/bowhead-viewer/BowheadWhaleDetector/bowhead/serve/app.py

# Update these lines:
# HTML_FILE = Path("/opt/bowhead-viewer/BowheadWhaleDetector/docs/eval_embedding_visualization.html")
# AUDIO_DIR = Path("/mnt/audio_files")  # ← Your actual .wav location
```

### Step 4: Install & Run
```bash
# One-time setup
bash /opt/bowhead-viewer/BowheadWhaleDetector/bowhead/serve/setup-orin.sh

# Start server
sudo systemctl start bowhead-viewer
sudo systemctl status bowhead-viewer
```

### Step 5: Expose Publicly (Cloudflare Tunnel)
```bash
# On Orin:
cloudflared tunnel login
cloudflared tunnel create bowhead-viewer
cloudflared tunnel route dns bowhead-viewer yourdomain.com

# Keep tunnel running:
cloudflared tunnel run bowhead-viewer

# Public URL: https://bowhead-viewer.yourdomain.com
```

---

## API Endpoints

### GET `/` - Main Viewer
Returns the interactive HTML embedding visualization with Plotly.js
```
http://localhost:8000/
```

### GET `/api/audio/{sample_id}` - Audio Streaming
Streams WAV file for a sample ID
```
curl http://localhost:8000/api/audio/sample_001
# Returns: binary WAV audio
```

### GET `/health` - Server Status
Returns server health and configuration info
```
curl http://localhost:8000/health
# Returns:
# {
#   "status": "ok",
#   "html_file": "/opt/...",
#   "audio_dir": "/mnt/...",
#   "audio_files_available": 2165
# }
```

---

## How It Works: User Clicks a Point

1. **User clicks scatter plot point** in browser
   - Plotly.js triggers click handler
   - JavaScript extracts `sample_id` from point metadata

2. **HTML requests audio from Flask API**
   ```javascript
   audioElement.src = "/api/audio/sample_id"
   ```

3. **Orin AGX Flask server receives request**
   - Looks up filename using `get_audio_filename(sample_id)`
   - Searches in `/mnt/audio_files/`
   - Streams binary WAV data back to browser

4. **Browser audio player displays and plays**
   - `<audio>` element loads the stream
   - User can press play/pause
   - Works from anywhere on the internet!

---

## Customization

### Different Audio Filename Format?
Edit `bowhead/serve/app.py`:

```python
def get_audio_filename(sample_id):
    # Current: sample_001.wav
    # return f"{sample_id}.wav"
    
    # Alternative: bowhead_20240715_001.wav
    # return f"bowhead_{sample_id}.wav"
    
    # Alternative: with folder structure
    # return f"site3/{sample_id}.wav"
```

### Using Large External Storage?
```bash
# Mount external drive
sudo mount /dev/sda1 /mnt/external_audio

# Update app.py:
# AUDIO_DIR = Path("/mnt/external_audio/audio_files")
```

### Restrict to Local Network Only?
Remove Cloudflare Tunnel step, use Tailscale instead:
```bash
sudo tailscale up
# URL: https://<orin-hostname>.ts.net:8000
# Only accessible from your devices
```

---

## Troubleshooting

### Audio files return 404
```bash
# Check actual files exist
ls /mnt/audio_files | head

# Check server logs
journalctl -u bowhead-viewer -f

# Test manual filename lookup
curl http://localhost:8000/api/audio/sample_001 -v
```

### HTML viewer doesn't load
```bash
# Verify HTML file exists
ls -lh /opt/bowhead-viewer/BowheadWhaleDetector/docs/

# Restart server
sudo systemctl restart bowhead-viewer
```

### Public URL works locally but not from internet
```bash
# Cloudflare Tunnel status
cloudflared tunnel status

# Restart tunnel
cloudflared tunnel run bowhead-viewer

# Check DNS
nslookup bowhead-viewer.yourdomain.com
```

---

## Performance Considerations

| Metric | Typical |
|--------|---------|
| HTML load time | 1-2 sec |
| Scatter plot render | 1-3 sec (2165 points) |
| Audio stream start | <200ms |
| Concurrent users | 8-12 (Orin 8-core) |
| Bandwidth per user | ~100 kB/sec (WAV stream) |

### Scaling
- **More users?** Add workers: `--workers 16` in systemd service
- **Slow audio?** Check Orin disk speed: `hdparm -t /dev/sd*`
- **Many audio files?** Use SSD or external fast storage

---

## Security Notes

- ✅ Cloudflare Tunnel provides SSL/TLS encryption
- ✅ No authentication required (modify if needed)
- ⚠️ Public endpoint = anyone can download audio files
- 💡 Optional: Add token-based auth to `/api/audio/` endpoint

---

## Next Steps

1. ✅ Create Orin AGX setup script (done)
2. ✅ Write Flask backend (done)
3. ⏳ **Generate embedding viewer HTML** (run on your Mac)
4. ⏳ **Copy files to Orin** (git push or SCP)
5. ⏳ **Run setup-orin.sh** (one-time)
6. ⏳ **Configure Cloudflare Tunnel** (10 minutes)
7. ⏳ **Test public access** (share link!)
