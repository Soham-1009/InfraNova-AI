# Deployment & Serving Documentation — InfraNova AI

**Document Version:** 3.2 (Production VM Deployment Runbook & Security Hardening)  
**Date:** 2026-09-14  
**Modules**: `api/`, `web/`, `Dockerfile`, `docker-compose.yml`, `requirements-prod.txt`, `.env.example`  

---

## 1. Architecture Overview

InfraNova AI runs as a single unified production container:

```
                      [ Internet / End Users ]
                                  │
                                  ▼
                         [ Port 80 / 443 ]
                   ┌─────────────────────────────┐
                   │  Reverse Proxy (Nginx/Caddy)│  <── Let's Encrypt SSL/TLS
                   └──────────────┬──────────────┘
                                  │ proxy_pass http://127.0.0.1:8000
                                  ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                   Docker Container: infranova-ai                │
 │                                                                 │
 │   FastAPI Backend (Port 8000) ─── Mounts ───► React SPA (/)     │
 │                │                                                │
 │                ▼                                                │
 │   InferenceEngine (Exp9 ResNet)                                 │
 └────────────────┼────────────────────────────────────────────────┘
                  │  Read-Only Volume Mount: -v ./outputs:/app/outputs:ro
                  ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ Host Private Storage: /opt/infranova-ai/outputs/best/           │
 │   - pix2pix_landsat_best.pth  (SHA-256: 71bbda3f...)            │
 └─────────────────────────────────────────────────────────────────┘
```

- **Unified Single-Origin Routing**: In production, the React frontend is pre-built into `web/dist` and mounted directly by FastAPI at `/`. API calls use relative paths (`/colorize`, `/health`), eliminating hardcoded hostnames and preventing HTTPS Mixed-Content blocking.
- **Vite Development Proxy**: In local development (`npm run dev`), the Vite dev server on port 5173 proxies `/colorize`, `/predict`, `/health`, `/thermal-preview`, and `/postprocess` directly to `http://localhost:8000`.
- **Private Checkpoint Governance**: The GitHub repository is public; trained model weights (`outputs/best/pix2pix_landsat_best.pth`) remain local and untracked. To maintain this boundary, the Docker image does **not** bake weights into the image; weights are mounted read-only (`./outputs:/app/outputs:ro`) at runtime.
- **Localhost Loopback Isolation**: In `docker-compose.yml`, the container port is bound strictly to `127.0.0.1:${PORT:-8000}:8000`. On Linux, Docker manipulates `iptables` directly and bypasses UFW by default; binding to `127.0.0.1` ensures port 8000 is never exposed to the public internet and can only be reached through the local reverse proxy.

---

## 2. Linux VM Production Deployment Runbook

Follow these steps to deploy InfraNova AI to an Ubuntu 22.04 / 24.04 LTS VM (AWS EC2, GCP Compute Engine, DigitalOcean, Oracle Cloud, Azure, or Hetzner).

### Step 1: Server Provisioning & Docker Installation

SSH into your server and prepare the environment:

```bash
# 1.1 Update package index and install prerequisites
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# 1.2 Prevent OOM killer on <= 4GB RAM VMs (Vite build + PyTorch wheel installation peak at ~2GB)
if [ $(free -m | awk '/^Mem:/{print $2}') -le 4096 ] && [ $(free -m | awk '/^Swap:/{print $2}') -eq 0 ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# 1.3 Add Docker's official GPG key (--yes avoids interactive prompt on re-runs)
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --yes --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# 1.4 Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 1.5 Install Docker Engine and Compose plugin
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 1.6 Grant non-root Docker execution
sudo usermod -aG docker $USER
# Note: Log out and log back in, or use 'sg docker' for non-interactive scripts
```

---

### Step 2: Clone Repository on Server

```bash
# Create application directory
sudo mkdir -p /opt/infranova-ai
sudo chown -R $USER:$USER /opt/infranova-ai
cd /opt/infranova-ai

# Clone the repository
git clone https://github.com/Soham-1009/InfraNova-AI.git .

# Create the outputs directory for mounting the private model weights
mkdir -p outputs/best
```

---

### Step 3: Secure Checkpoint Transfer & Verification

From your **local machine**, transfer the private Exp9 production weights to the server using SCP:

```powershell
# Windows PowerShell (Local Machine)
# Replace <SERVER_IP> and include -i <key.pem> if using key-based cloud authentication
scp -i "C:\path\to\your-key.pem" outputs/best/pix2pix_landsat_best.pth ubuntu@<SERVER_IP>:/opt/infranova-ai/outputs/best/
```

On the **server**, verify the SHA-256 checksum and file permissions:

```bash
cd /opt/infranova-ai

# Verify checksum matches the official Exp9 production release
sha256sum outputs/best/pix2pix_landsat_best.pth
```

> [!IMPORTANT]
> The computed hash must strictly match:  
> `71bbda3f31b85e7e741b26d5ce0ff398a0394c7dd4ef6ff5452f31f7e0400382`

```bash
# Set read-only permissions for the container's unprivileged appuser (UID 1000)
chmod 644 outputs/best/pix2pix_landsat_best.pth
```

---

### Step 4: Environment Configuration

Create the production `.env` configuration on the server:

```bash
cd /opt/infranova-ai
cp .env.example .env
```

The default `.env` configuration:
```ini
# Port exposed on host loopback 127.0.0.1 (default: 8000)
PORT=8000

# Path to the mounted model inside container
INFRANOVA_CHECKPOINT=/app/outputs/best/pix2pix_landsat_best.pth

# Inference execution device ("cpu" or "cuda", blank for auto-detection)
INFRANOVA_DEVICE=

# Spatial tile size (default: 128)
INFRANOVA_IMAGE_SIZE=128

# Allowed CORS origins
INFRANOVA_CORS_ORIGINS=*

# Frontend relative API routing (leave blank for same-origin production)
VITE_API_URL=
```

---

### Step 5: Firewall Configuration (`ufw`)

Configure UFW so only SSH (22), HTTP (80), and HTTPS (443) are publicly reachable:

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable
sudo ufw status verbose
```

> [!NOTE]
> Because `docker-compose.yml` binds to `127.0.0.1:${PORT:-8000}:8000`, Docker is forced to listen exclusively on loopback. Port 8000 is inaccessible from the internet and can only be routed through your reverse proxy.

---

### Step 6: Build & Start Container with Docker Compose

```bash
cd /opt/infranova-ai

# Build the production container (compiles React bundle in Node 20, sets up Python 3.11 runtime)
# and starts the service in detached mode
docker compose up --build -d
```

Verify container status:
```bash
# Check container health (status will report 'healthy' after probe passes)
docker compose ps

# Follow logs if needed
docker compose logs -f
```

Test the local endpoint directly on the server:
```bash
curl -i http://127.0.0.1:8000/health
```

Expected output:
```json
HTTP/1.1 200 OK
content-type: application/json

{"status":"ok","model_loaded":true,"device":"cpu"}
```

---

### Step 7: Domain & HTTPS Reverse Proxy Setup

Set up a reverse proxy to terminate SSL/TLS and forward requests to `http://127.0.0.1:8000`.

#### Option A: Caddy (Recommended — Automatic HTTPS)

```bash
# Install Caddy
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLF 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --yes --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLF 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update
sudo apt-get install -y caddy

# Configure /etc/caddy/Caddyfile
sudo tee /etc/caddy/Caddyfile > /dev/null << 'EOF'
your-domain.com {
    reverse_proxy 127.0.0.1:8000

    request_body {
        max_size 50MB
    }
}
EOF

# Restart Caddy
sudo systemctl restart caddy
```

#### Option B: Nginx + Certbot (Standard Enterprise)

```bash
# Install Nginx and Certbot
sudo apt-get install -y nginx certbot python3-certbot-nginx

# Configure /etc/nginx/sites-available/infranova
sudo tee /etc/nginx/sites-available/infranova > /dev/null << 'EOF'
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Disable buffering for streaming raster responses
        proxy_buffering off;
        proxy_read_timeout 120s;
    }
}
EOF

# Enable configuration and remove default placeholder
sudo ln -sf /etc/nginx/sites-available/infranova /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test configuration and restart Nginx
sudo nginx -t
sudo systemctl restart nginx

# Obtain and install Let's Encrypt SSL certificate
sudo certbot --nginx -d your-domain.com --non-interactive --agree-tos --email admin@your-domain.com --redirect
```

---

### Step 8: ARM64 & Oracle Cloud Ampere A1 Guidelines

InfraNova AI runs natively on ARM64 (`aarch64`) instances such as Oracle Cloud Ampere A1, AWS Graviton, and Apple Silicon.

1. **Kernel Page Size Check**:
   ```bash
   getconf PAGESIZE
   ```
   - Must return `4096` (4KB).
   - If running on Oracle Linux with a 64KB kernel (`kernel-uek64k` returning `65536`), memory allocators will crash. Boot into the standard 4KB kernel or deploy Ubuntu 22.04 / 24.04 LTS.
2. **OpenMP CPU Thread Limit**:
   Ampere A1 instances provide up to 80 physical cores. To prevent thread synchronization thrashing on small image tiles, set `OMP_NUM_THREADS=4` in `.env`.
3. **Build Architecture**:
   Build the container image directly on the ARM64 instance, or use Docker Buildx (`docker buildx build --platform linux/arm64`) to avoid QEMU emulation faults (`Illegal instruction`).

---

## 3. Endpoints Reference

| Endpoint | Method | Input Parameters | Output Response | Function |
| :--- | :---: | :--- | :--- | :--- |
| **`/health`** | `GET` | None | JSON `{"status": "ok", "model_loaded": bool, "device": str}` | Liveness & model status probe |
| **`/colorize`** | `POST` | `file`: UploadFile (`.tif`, `.tiff`, `.png`, `.npy`), `tta`: bool | Streaming PNG (`image/png`) | Synthesizes true color RGB image |
| **`/thermal-preview`** | `POST` | `file`: UploadFile | Streaming PNG (`image/png`) | Renders thermal Inferno colormap |
| **`/postprocess/clahe`**| `POST`| `file`: UploadFile, `clip_limit`: float (2.0), `grid_size`: int (8) | Streaming PNG (`image/png`) | LAB lightness contrast enhancement |

---

## 4. Model Checkpoint Governance & Rollback Ledger

- **Active Production Checkpoint**:
  - Path: `outputs/best/pix2pix_landsat_best.pth`
  - SHA-256: `71bbda3f31b85e7e741b26d5ce0ff398a0394c7dd4ef6ff5452f31f7e0400382`
  - Architecture: `Pix2PixHDGlobalResNetGenerator` (11,369,795 parameters)
  - Key Metrics: 13.745 dB PSNR, 0.4502 SSIM, 0.1723 MAE, 0.1968 rad SAM (11.28°), 0.3711 YOLOv8n F1@0.25
- **Archived Legacy Production Backup**:
  - Path: `outputs/best/pix2pix_landsat_backup_20260912_231938.pth`
  - SHA-256: `4604d36d07a0fb4c0696a53040c17004084068c51cc745a23f69b76c8baf6aa8`
  - Architecture: `Pix2PixHDGenerator` (21,383,238 parameters)
- **Rollback Metadata**: Maintained at `outputs/best/rollback_metadata.json` for one-command rollback capability.
