"""Shared skeleton-geometry primitives — single source of truth.

Both the zero-training heuristics (`detect/heuristics.py`) and the trained-model
feature extractor (`detect/skeleton_features.py`) need the exact same geometry:
torso horizontalness, box IoU, inter-person proximity, and the aggressive-pose
cue. Keeping one implementation here prevents the two paths from silently
drifting apart — a drift that would make the trained model see a different
feature distribution at inference than the heuristic fallback computes.

These functions are pure and numerically identical to the inline versions they
replaced, so `skeleton_features.frame_feature()` output is unchanged (trained
weights stay valid).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .pose import (L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
                   R_ANKLE, R_HIP, R_KNEE, R_SHOULDER, R_WRIST, PersonDetection)


def mid(a, b):
    """Midpoint of two (x, y) points; None if either is missing."""
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-6))


def proximity(p1: PersonDetection, p2: PersonDetection) -> float:
    """1 when two people are essentially on top of each other, decaying with distance."""
    dist = float(np.linalg.norm(p1.centroid - p2.centroid))
    scale = (p1.height + p2.height) / 2.0 + 1e-6
    return float(np.exp(-dist / scale))


def scale_ratio(p1: PersonDetection, p2: PersonDetection) -> float:
    """Ratio of smaller to larger person height in (0, 1]."""
    min_h = min(p1.height, p2.height)
    max_h = max(p1.height, p2.height) + 1e-6
    return float(min_h / max_h)


def foot_distance_ratio(p1: PersonDetection, p2: PersonDetection) -> float:
    """Vertical difference in ground-contact (bottom of box) relative to height."""
    max_h = max(p1.height, p2.height) + 1e-6
    return float(abs(p1.box[3] - p2.box[3]) / max_h)


def are_depth_compatible(p1: PersonDetection, p2: PersonDetection,
                         min_scale_ratio: float = 0.58) -> bool:
    """Return True if two people share approximately the same depth plane.

    In perspective projection, objects at different depths (Z-axis) have different
    apparent heights and ground contact Y positions. Two adults physically interacting
    on the ground will have comparable height (scale ratio >= 0.58) and feet roughly
    level with the ground plane (foot_diff <= 0.22, relaxed if heights are very close).
    """
    sr = scale_ratio(p1, p2)
    foot_diff = foot_distance_ratio(p1, p2)
    return (sr >= min_scale_ratio) and (foot_diff <= 0.22 or sr >= 0.75)


def box_gap(p1: PersonDetection, p2: PersonDetection) -> float:
    """Euclidean distance between the closest edges of two bounding boxes (0 if overlapping)."""
    dx = max(0.0, max(p1.box[0] - p2.box[2], p2.box[0] - p1.box[2]))
    dy = max(0.0, max(p1.box[1] - p2.box[3], p2.box[1] - p1.box[3]))
    return float(math.hypot(dx, dy))


def are_in_combat_range(p1: PersonDetection, p2: PersonDetection) -> bool:
    """Return True if two people are physically capable of hand-to-hand or striking contact.

    Must be depth-compatible (not a distant background spectator) and within reach:
    either overlapping (IoU >= 0.05) or separated by less than arm/leg reach
    (box edge gap <= 0.55 * scale and centroid dist <= 1.25 * scale).
    """
    if not are_depth_compatible(p1, p2):
        return False
    scale = (p1.height + p2.height) / 2.0 + 1e-6
    dist = float(np.linalg.norm(p1.centroid - p2.centroid))
    gap = box_gap(p1, p2)
    iou = box_iou(p1.box, p2.box)
    return (iou >= 0.05) or ((gap <= 0.55 * scale) and (dist <= 1.25 * scale))


def torso_horizontalness(sh, hip) -> Optional[float]:
    """0 = upright, 1 = fully horizontal, from the shoulder->hip torso vector.

    `sh` and `hip` are the shoulder- and hip-midpoint (x, y) arrays (as produced
    by `mid`). Returns None when either is missing or the torso is degenerate.
    """
    if sh is None or hip is None:
        return None
    v = hip - sh
    if np.linalg.norm(v) < 1e-3:
        return None
    angle_from_vertical = abs(math.degrees(math.atan2(abs(v[0]), abs(v[1]) + 1e-6)))
    return min(1.0, angle_from_vertical / 90.0)


def leg_horizontalness(p: PersonDetection) -> Optional[float]:
    """0 = vertical/standing legs (perpendicular to floor), 1 = lying flat (parallel to floor).

    Evaluates the hip->ankle (and hip->knee) leg axes against the floor plane.
    Returns None if no confident leg keypoints are available.
    """
    angles = []
    for hip_idx, knee_idx, ankle_idx in ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)):
        hip = p.kp(hip_idx, min_conf=0.20)
        knee = p.kp(knee_idx, min_conf=0.20)
        ankle = p.kp(ankle_idx, min_conf=0.20)

        # Ignore uninitialized (0, 0) coordinates
        if hip is not None and np.linalg.norm(hip) < 1.0:
            hip = None
        if knee is not None and np.linalg.norm(knee) < 1.0:
            knee = None
        if ankle is not None and np.linalg.norm(ankle) < 1.0:
            ankle = None

        seg_angles = []
        for a, b in ((hip, ankle), (hip, knee), (knee, ankle)):
            if a is not None and b is not None:
                v = b - a
                if np.linalg.norm(v) > 1e-3:
                    deg = abs(math.degrees(math.atan2(abs(v[0]), abs(v[1]) + 1e-6)))
                    seg_angles.append(min(1.0, deg / 90.0))
        if seg_angles:
            angles.append(seg_angles[0])

    if not angles:
        return None
    return float(np.mean(angles))


def aggressive_pose(p: PersonDetection) -> float:
    """Wrists raised near/above shoulder height — a weak aggression/defense cue."""
    # A person lying flat on the floor (horizontal posture) has arms level with shoulders
    # by geometry, but is not in an aggressive combat stance.
    aspect = p.width / (p.height + 1e-6)
    if aspect >= 1.25:
        return 0.0
    leg_h = leg_horizontalness(p)
    if leg_h is not None and leg_h >= 0.45:
        return 0.0
    sh = mid(p.kp(L_SHOULDER), p.kp(R_SHOULDER))
    hip = mid(p.kp(L_HIP), p.kp(R_HIP))
    torso_h = torso_horizontalness(sh, hip)
    if torso_h is not None and torso_h >= 0.50:
        return 0.0

    cues = 0
    n = 0
    for wi, si in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
        w, s = p.kp(wi), p.kp(si)
        if w is not None and s is not None:
            n += 1
            if w[1] <= s[1] + 0.02 * p.height:   # wrist at/above shoulder level (image y grows down)
                cues += 1
    return (cues / n) if n else 0.0

