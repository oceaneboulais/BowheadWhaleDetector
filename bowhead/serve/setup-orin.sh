#!/bin/bash
# Deployment script for Bowhead Viewer on NVIDIA Orin AGX
# Run this on the Orin to set up the server

set -e  # Exit on error

echo "========================================"
echo "Bowhead Viewer - Orin AGX Setup"
echo "========================================"

# 1. Update system
echo "[1/6] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# 2. Install Python + dependencies
echo "[2/6] Installing Python and dependencies..."
sudo apt install -y python3-pip python3-venv

# 3. Create app directory
APP_DIR="/opt/bowhead-viewer"
echo "[3/6] Creating app directory at $APP_DIR..."
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR

# 4. Clone/copy repo (if not already present)
if [ ! -d "$APP_DIR/BowheadWhaleDetector" ]; then
    echo "[4/6] Cloning repository..."
    cd $APP_DIR
    git clone https://github.com/oceaneboulais/BowheadWhaleDetector.git
    cd BowheadWhaleDetector
else
    echo "[4/6] Repository already present, pulling latest..."
    cd $APP_DIR/BowheadWhaleDetector
    git pull
fi

# 5. Install Python dependencies
echo "[5/6] Installing Python dependencies..."
pip install -q flask gunicorn

# 6. Install Cloudflare Tunnel (optional but recommended)
echo "[6/6] Installing Cloudflare Tunnel for public access..."
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared.deb
rm cloudflared.deb

echo ""
echo "========================================"
echo "Setup complete! Next steps:"
echo "========================================"
echo ""
echo "1. Configure paths in bowhead/serve/app.py:"
echo "   - Update AUDIO_DIR to your .wav files location"
echo "   - Update HTML_FILE path"
echo ""
echo "2. For public access via Cloudflare Tunnel:"
echo "   cloudflared tunnel login"
echo "   cloudflared tunnel create bowhead-viewer"
echo "   cloudflared tunnel route dns bowhead-viewer <your-domain.com>"
echo ""
echo "3. Start the server:"
echo "   cd $APP_DIR/BowheadWhaleDetector"
echo "   gunicorn --workers 4 --bind 0.0.0.0:8000 --timeout 300 bowhead.serve.app:app"
echo ""
echo "4. (Optional) Install as systemd service:"
echo "   sudo cp bowhead/serve/bowhead-viewer.service /etc/systemd/system/"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable bowhead-viewer"
echo "   sudo systemctl start bowhead-viewer"
echo ""
