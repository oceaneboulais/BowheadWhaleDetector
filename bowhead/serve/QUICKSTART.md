# NVIDIA Orin AGX Hosting: Quick Start

## Your Complete System (Tested ✓)

```
MacBook (your dev machine)
    ↓ Generate HTML viewer
    ↓ Push to GitHub
    │
NVIDIA Orin AGX (at your lab)
    ├─ Flask backend (bowhead/serve/app.py)
    ├─ Static HTML (docs/eval_embedding_visualization.html)
    ├─ Local .wav files (/mnt/audio_files/)
    └─ Cloudflare Tunnel → Public Internet
    │
Browser (anywhere on Earth)
    → https://bowhead-viewer.yourdomain.com
    → Click scatter point → see thumbnail + stream audio
```

---

## Step 1️⃣: Generate HTML on Your Mac (Right Now)

```bash
cd /Users/oceaneboulais/Github/BowheadWhaleDetector

# For local-only testing (file:// URIs):
python -m bowhead.eval.build_embedding_visualization \
  --data data/spectrograms_eval_v2.npz \
  --out docs/eval_embedding_visualization.html

# OR for Orin AGX with Flask API (recommended for production):
python -m bowhead.eval.build_embedding_visualization \
  --data data/spectrograms_eval_v2.npz \
  --out docs/eval_embedding_visualization.html \
  --api-audio-base "http://localhost:8000/api/audio"
```

**Output:** `docs/eval_embedding_visualization.html` (self-contained, all data embedded)

---

## Step 2️⃣: Push to GitHub

```bash
# Commit the server code + HTML viewer
git add bowhead/serve/
git add docs/eval_embedding_visualization.html
git commit -m "Add Orin AGX Flask server + interactive embedding viewer"
git push origin add-results-dashboards
```

---

## Step 3️⃣: Set Up Orin AGX (One Time)

**SSH into your Orin:**
```bash
ssh orin@<orin-ip>
```

**Run setup script:**
```bash
cd /tmp
curl -O https://raw.githubusercontent.com/oceaneboulais/BowheadWhaleDetector/add-results-dashboards/bowhead/serve/setup-orin.sh
chmod +x setup-orin.sh
./setup-orin.sh
```

**Configure paths in the Flask app:**
```bash
nano /opt/bowhead-viewer/BowheadWhaleDetector/bowhead/serve/app.py
```

Update these lines with your actual paths:
```python
# Line 15-16:
HTML_FILE = Path("/opt/bowhead-viewer/BowheadWhaleDetector/docs/eval_embedding_visualization.html")
AUDIO_DIR = Path("/mnt/audio_files")  # ← Your .wav files location
```

Save & exit (Ctrl+X, Y, Enter)

---

## Step 4️⃣: Start the Server

**Quick test (local only):**
```bash
cd /opt/bowhead-viewer/BowheadWhaleDetector
gunicorn --workers 4 --bind 0.0.0.0:8000 --timeout 300 bowhead.serve.app:app
```

Visit: `http://localhost:8000/`

**Install as background service (auto-restart):**
```bash
sudo cp /opt/bowhead-viewer/BowheadWhaleDetector/bowhead/serve/bowhead-viewer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bowhead-viewer
sudo systemctl start bowhead-viewer
sudo systemctl status bowhead-viewer
```

---

## Step 5️⃣: Expose to Public Internet (Cloudflare Tunnel)

**On Orin:**
```bash
# Authenticate with Cloudflare
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create bowhead-viewer

# Route to your domain
cloudflared tunnel route dns bowhead-viewer yourdomain.com

# Create config file
mkdir -p ~/.cloudflared
cat > ~/.cloudflared/config.yml <<'EOF'
tunnel: bowhead-viewer
credentials-file: ~/.cloudflared/YOUR-TUNNEL-UUID.json

ingress:
  - hostname: bowhead-viewer.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
EOF

# Start tunnel (keep running)
cloudflared tunnel run bowhead-viewer
```

**Your public URL:**
```
https://bowhead-viewer.yourdomain.com
```

Share this link with anyone! Works from anywhere on Earth 🌍

---

## Step 6️⃣: Optional - Systemd Service for Tunnel

If you want the tunnel to auto-restart too:

```bash
sudo cat > /etc/systemd/system/cloudflare-tunnel.service <<'EOF'
[Unit]
Description=Cloudflare Tunnel for Bowhead Viewer
After=network.target

[Service]
Type=simple
User=orin
ExecStart=/usr/bin/cloudflared tunnel run bowhead-viewer
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable cloudflare-tunnel
sudo systemctl start cloudflare-tunnel
```

---

## Testing Checklist

- [ ] HTML viewer loads at `http://localhost:8000/`
- [ ] Can see scatter plot with 2165 points
- [ ] Click on a point → see thumbnail + metadata
- [ ] Audio player appears when you click a point
- [ ] Click audio play button → .wav file streams
- [ ] Public URL works from your phone/laptop elsewhere
- [ ] Scroll/zoom scatter plot is responsive

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **404: Audio not found** | Check filename format in `get_audio_filename()`, verify files in `/mnt/audio_files/` |
| **HTML blank** | Check `HTML_FILE` path in app.py, restart with `sudo systemctl restart bowhead-viewer` |
| **Public URL doesn't work** | Check tunnel: `cloudflared tunnel status`, re-run `cloudflared tunnel run bowhead-viewer` |
| **Server slow** | Increase workers: edit systemd service, set `--workers 8` |
| **Port 8000 in use** | `lsof -i :8000`, or use different port in app.py |

---

## Summary: What You Get

✅ **Interactive embedding viewer** (UMAP + PaCMAP tabs)
✅ **Real .wav audio streaming** from Orin's local disk  
✅ **Public HTTPS URL** via Cloudflare Tunnel  
✅ **Works from anywhere** on the internet  
✅ **No external storage needed** (all on Orin)  
✅ **Zero file size limits** (keep all audio on device)  
✅ **Fast playback** (LAN speeds)  

**Total setup time:** ~30 minutes (mostly waiting for downloads)  
**Monthly cost:** $0 (you already have the Orin!)
