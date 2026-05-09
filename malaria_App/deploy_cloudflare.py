import os
import subprocess
import sys
import time

print("=" * 60)
print("  Malaria-AI  |  Cloudflare Tunnel Launcher")
print("=" * 60)

# ─── 1. Download cloudflared binary ───────────────────────────────────────────
CLOUDFLARED = "./cloudflared"

if not os.path.exists(CLOUDFLARED):
    print("\n[1/4] Downloading Cloudflare engine...")
    os.system(
        "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared"
    )
    print("      ✅ cloudflared downloaded and made executable.")
else:
    print("\n[1/4] cloudflared already present — skipping download.")

# ─── 2. Install Python dependencies ───────────────────────────────────────────
print("\n[2/4] Installing Python dependencies...")
os.system(
    "pip install -q streamlit tensorflow pillow numpy h5py"
)
print("      ✅ Dependencies installed.")

# ─── 3. Launch Streamlit in the background ────────────────────────────────────
print("\n[3/4] Launching Streamlit server (port 8501)...")

log_path = "streamlit_logs.txt"
with open(log_path, "w") as log_file:
    subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port",              "8501",
            "--server.address",           "0.0.0.0",
            "--server.headless",          "true",
            "--server.enableCORS",        "false",
            "--server.enableXsrfProtection", "false",
        ],
        stdout=log_file,
        stderr=log_file,
    )

print(f"      ✅ Streamlit started. Logs → {log_path}")
print("      Waiting 6 seconds for the server to warm up…")
time.sleep(6)

# ─── 4. Start Cloudflare Tunnel ───────────────────────────────────────────────
print("\n[4/4] Opening Cloudflare Tunnel…")
print("-" * 60)
print("  👇  CLICK THE LINK BELOW TO OPEN YOUR APP  👇")
print("-" * 60)

# This blocks until the tunnel is closed (Ctrl+C or kernel interrupt)
os.system(f"{CLOUDFLARED} tunnel --url http://localhost:8501")
