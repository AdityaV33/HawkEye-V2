# HawkEye — Setup & Deployment Guide

## 1. System Requirements
- **OS**: Linux (Ubuntu 20.04+, Debian, Arch), macOS (Apple Silicon M1/M2/M3/M4 or Intel), or Windows 10/11 (WSL2 or Native).
- **Python**: 3.10, 3.11, or 3.12.
- **Hardware Acceleration**:
  - NVIDIA GPU (CUDA 12.x) with RTX 30/40/50 series.
  - Apple Silicon GPU (Metal/MPS).
  - CPU mode (graceful fallback on standard laptops).

---

## 2. Installation

### On Linux (NVIDIA CUDA) / Windows
```bash
python3 -m venv .venv
source .venv/bin/activate

# 1. Install PyTorch with matching CUDA build
pip install "torch>=2.4" torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Install remaining dependencies
pip install -r requirements-cuda.txt
```

### On macOS / Standard CPU
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Warm Pretrained Weights (Optional)
```bash
python scripts/download_models.py
```
*(Pre-downloads YOLO11n-pose and InsightFace face recognition pack so the first run starts instantly).*

---

## 3. Running HawkEye

```bash
# Start with default webcam:
python -m hawkeye.run

# Start with desktop screen capture:
python -m hawkeye.run --source screen

# Loop a video clip:
python -m hawkeye.run --source clip.mp4

# Run with HTTPS (enables browser camera access from mobile phones):
python -m hawkeye.run --ssl
```

Open **`http://localhost:8000`** in your browser. The unified FastAPI server serves the API, WebSocket stream, and operator web app together on one port.

---

## 4. Reasoner Backends (Stage B VLM)

| Backend | Requirements | Notes |
|---|---|---|
| `mock` | None (built-in) | Instant templated explanations. Zero cost, ideal for testing. |
| `claude` | `export ANTHROPIC_API_KEY="..."` | Multimodal Claude 3.5 Sonnet. Best explanations and detailed incident summaries. |
| `moondream` | `pip install transformers accelerate einops pillow` | Fully offline local VLM. Runs on edge GPUs / Apple Silicon. |

`backend: auto` automatically picks Claude if `ANTHROPIC_API_KEY` is set, otherwise falls back to `mock`.

---

## 5. Camera & Video Ingest Options

- **Webcam**: `--source 0`
- **Desktop Screen**: `--source screen`
- **Wireless Phone Camera**: Open `https://<laptop-ip>:8000` on your phone browser, go to the **Camera** tab, and press **Broadcast to HawkEye**.
- **IP Camera / RTSP**: `--source http://<phone-ip>:8080/video` or `--source rtsp://<stream-url>`
