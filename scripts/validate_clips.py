"""Validate video clips in the action dataset — find corrupt or too-short files.

Scans every video in <root>/<class>/ and reports:
  - Files that can't be opened by OpenCV
  - Files with fewer than --min-frames usable frames
  - Files with zero dimensions or degenerate resolution

Usage:
    python scripts/validate_clips.py --data D:/datasets/actions --min-frames 16
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def validate_clip(path: Path, min_frames: int) -> dict | None:
    """Return an issue dict if the clip is bad, else None."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return {"file": str(path), "issue": "cannot_open", "frames": 0, "resolution": "N/A"}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w == 0 or h == 0:
        cap.release()
        return {"file": str(path), "issue": "zero_resolution", "frames": 0,
                "resolution": f"{w}x{h}"}

    count = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        count += 1
        if count >= min_frames:
            # We've verified enough frames — no need to decode the whole clip
            # just for validation. Get total from metadata.
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > count:
                count = total
            break
    cap.release()

    if count < min_frames:
        return {"file": str(path), "issue": f"too_short ({count} frames)",
                "frames": count, "resolution": f"{w}x{h}"}

    return None


def main():
    ap = argparse.ArgumentParser(description="Validate action dataset clips")
    ap.add_argument("--data", required=True, help="Root of action dataset (e.g. D:\\datasets\\actions)")
    ap.add_argument("--min-frames", type=int, default=16, dest="min_frames",
                     help="Minimum usable frames (default: 16)")
    ap.add_argument("--quarantine", action="store_true",
                     help="Move bad clips to <root>/_quarantine/<class>/ instead of just reporting")
    args = ap.parse_args()

    root = Path(args.data)
    if not root.exists():
        sys.exit(f"Dataset root not found: {root}")

    classes = sorted([d.name for d in root.iterdir()
                      if d.is_dir() and not d.name.startswith("_")])
    print(f"Classes: {classes}")
    print(f"Min frames: {args.min_frames}")
    print()

    issues: list[dict] = []
    total_scanned = 0

    for cls in classes:
        cls_dir = root / cls
        clips = sorted([f for f in cls_dir.rglob("*")
                        if f.is_file() and f.suffix.lower() in VIDEO_EXT])
        print(f"  [{cls}] scanning {len(clips)} clips...", end="", flush=True)
        cls_issues = 0

        for i, clip in enumerate(clips):
            issue = validate_clip(clip, args.min_frames)
            if issue:
                issue["class"] = cls
                issues.append(issue)
                cls_issues += 1
            total_scanned += 1
            if (i + 1) % 50 == 0:
                print(f" {i+1}", end="", flush=True)

        status = f" [FAIL] {cls_issues} issues" if cls_issues else " [OK] all good"
        print(f" - {len(clips)} scanned{status}")

    print(f"\n{'='*60}")
    print(f"Total scanned: {total_scanned}")
    print(f"Issues found: {len(issues)}")

    if issues:
        print(f"\nProblematic clips:")
        for iss in issues:
            print(f"  [{iss.get('class', '?')}] {iss['issue']}: {iss['file']}")

        if args.quarantine:
            quarantine_dir = root / "_quarantine"
            moved = 0
            for iss in issues:
                src = Path(iss["file"])
                cls = iss.get("class", "unknown")
                dst = quarantine_dir / cls / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    src.rename(dst)
                    moved += 1
                    print(f"  Moved: {src.name} → _quarantine/{cls}/")
                except Exception as e:
                    print(f"  Failed to move {src.name}: {e}")
            print(f"\nMoved {moved} clips to {quarantine_dir}")
        else:
            print("\nRun with --quarantine to move bad clips out of the dataset.")
    else:
        print("\n✓ All clips are valid!")


if __name__ == "__main__":
    main()
