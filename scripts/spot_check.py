"""Spot-check dataset labels — generate an HTML contact sheet for visual review.

Randomly samples N clips per class, extracts a middle frame from each, and
produces an HTML file you can open in a browser to visually scan for mislabels.

Usage:
    python scripts/spot_check.py --data D:/datasets/actions --sample 20
    # Then open reports/spot_check.html in your browser
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract_middle_frame(video_path: str, target_size: tuple[int, int] = (320, 180)) -> str | None:
    """Extract the middle frame of a video, resize, and return as base64 JPEG."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Try reading frames to find midpoint
        frames = []
        while len(frames) < 300:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            return None
        frame = frames[len(frames) // 2]
    else:
        mid = total // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None

    frame = cv2.resize(frame, target_size)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def main():
    ap = argparse.ArgumentParser(description="Visual spot-check of dataset labels")
    ap.add_argument("--data", required=True, help="Root of action dataset")
    ap.add_argument("--sample", type=int, default=20,
                     help="Number of clips to sample per class (default: 20)")
    ap.add_argument("--output", default="reports/spot_check.html",
                     help="Output HTML file (default: reports/spot_check.html)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.data)
    if not root.exists():
        sys.exit(f"Dataset root not found: {root}")

    classes = sorted([d.name for d in root.iterdir()
                      if d.is_dir() and not d.name.startswith("_")])
    print(f"Classes: {classes}")

    rng = np.random.default_rng(args.seed)

    html_parts = ["""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>HawkEye Dataset Spot Check</title>
<style>
  body { font-family: Inter, sans-serif; background: #1a1a2e; color: #e0e0e0; margin: 20px; }
  h1 { color: #7c3aed; text-align: center; }
  h2 { color: #a78bfa; border-bottom: 1px solid #333; padding-bottom: 8px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, 340px); gap: 12px; }
  .card { background: #16213e; border-radius: 8px; padding: 8px; text-align: center; }
  .card img { width: 320px; height: 180px; border-radius: 4px; }
  .card .name { font-size: 11px; color: #888; margin-top: 4px; word-break: break-all; }
  .mislabel { color: #ef4444; font-weight: bold; cursor: pointer; }
  .stats { background: #16213e; padding: 12px 20px; border-radius: 8px; margin-bottom: 20px; }
</style>
</head><body>
<h1>🦅 HawkEye Dataset Spot Check</h1>
<div class="stats">
  <p>Review the thumbnails below. Look for clips that are clearly in the <strong>wrong class</strong>.</p>
  <p>Common mislabels to watch for:</p>
  <ul>
    <li><strong>fall/</strong> — clips showing someone sitting down, bending over, or lying still (not actually falling)</li>
    <li><strong>violence/</strong> — clips of sports, rough play, or vigorous exercise (not actual fighting)</li>
    <li><strong>normal/</strong> — clips containing obvious fights or falls</li>
  </ul>
</div>
"""]

    for cls in classes:
        cls_dir = root / cls
        clips = sorted([f for f in cls_dir.rglob("*")
                        if f.is_file() and f.suffix.lower() in VIDEO_EXT])
        n = min(args.sample, len(clips))
        sampled = rng.choice(len(clips), size=n, replace=False) if len(clips) > n else range(len(clips))

        html_parts.append(f'<h2>{cls}/ — {len(clips)} total clips (showing {n})</h2>\n<div class="grid">\n')

        for idx in sorted(sampled):
            clip = clips[idx]
            print(f"  Extracting frame from [{cls}] {clip.name}...", end="\r", flush=True)
            b64 = extract_middle_frame(str(clip))
            if b64:
                html_parts.append(f'''<div class="card">
  <img src="data:image/jpeg;base64,{b64}" alt="{clip.name}">
  <div class="name">{clip.name}</div>
</div>\n''')
            else:
                html_parts.append(f'''<div class="card">
  <div style="width:320px;height:180px;background:#333;display:flex;align-items:center;justify-content:center;border-radius:4px;color:#ef4444;">FAILED</div>
  <div class="name">{clip.name}</div>
</div>\n''')

        html_parts.append('</div>\n')
        print(f"  [{cls}] {n} thumbnails extracted" + " " * 40)

    html_parts.append("</body></html>")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(html_parts), encoding="utf-8")
    print(f"\n[OK] Spot-check report saved to: {out_path}")
    print(f"  Open it in your browser to review.")


if __name__ == "__main__":
    main()
