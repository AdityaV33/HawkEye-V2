"""Zero-training keypoint heuristics for fall and fight/violence.

These are an honest *bootstrap*: geometry + motion that works on day one so the
whole pipeline is demoable immediately. They are designed to be swapped for the
trained ST-GCN skeleton classifier (scripts/train_action.py) behind the same
`analyze()` return contract: List[(AnomalyType, score, reason)].
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..events.schema import AnomalyType
from . import geometry
from .pose import (L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, PersonDetection)

Candidate = Tuple[AnomalyType, float, str]


def is_human(p: PersonDetection) -> bool:
    """Verify that a detection has human anatomical structure (not background clutter)."""
    # Count keypoints with confidence >= 0.20
    kpts_conf = sum(1 for kp in p.keypoints if kp[2] >= 0.20)
    if kpts_conf < 3:
        return False
    # Must have at least one upper-body point (head/shoulders/arms: indices 0-10)
    # and at least one lower-body point (hips/knees/ankles: indices 11-16)
    has_upper = any(p.kp(i, min_conf=0.20) is not None for i in range(11))
    has_lower = any(p.kp(i, min_conf=0.20) is not None for i in range(11, 17))
    return has_upper and has_lower


def torso_points(p: PersonDetection):
    """Get shoulder and hip points, gracefully handling single-sided occlusion when a person is down."""
    # Shoulders
    ls = p.kp(L_SHOULDER, min_conf=0.20)
    rs = p.kp(R_SHOULDER, min_conf=0.20)
    if ls is not None and rs is not None:
        sh = (ls + rs) / 2.0
    elif ls is not None:
        sh = ls
    elif rs is not None:
        sh = rs
    else:
        # Fallback to nose if shoulders are occluded
        sh = p.kp(0, min_conf=0.20)

    # Hips
    lh = p.kp(L_HIP, min_conf=0.20)
    rh = p.kp(R_HIP, min_conf=0.20)
    if lh is not None and rh is not None:
        hip = (lh + rh) / 2.0
    elif lh is not None:
        hip = lh
    elif rh is not None:
        hip = rh
    else:
        # Fallback to knees if hips are occluded
        lk = p.kp(13, min_conf=0.20)
        rk = p.kp(14, min_conf=0.20)
        hip = (lk + rk) / 2.0 if (lk is not None and rk is not None) else (lk if lk is not None else rk)

    return sh, hip


def fall_score(p: PersonDetection) -> float:
    """Single-frame fall/lying likelihood in [0, 1].
    
    Guards strictly against non-human detections while gracefully handling
    one-sided limb occlusion during real collapses.
    """
    if not is_human(p):
        return 0.0

    sh, hip = torso_points(p)
    horiz = geometry.torso_horizontalness(sh, hip)
    aspect = p.width / (p.height + 1e-6)          # >1 => wider than tall (lying)
    aspect_term = np.clip((aspect - 0.75) / 1.1, 0.0, 1.0)

    if horiz is None:
        # If human torso vector is fully obscured, use human-confirmed aspect ratio
        return float(aspect_term * 0.65) if aspect >= 1.0 else 0.0

    # If the person is standing tall (height > width), slight/moderate
    # forward or lateral bending (angle < 43 deg, horiz < 0.48) is NOT a fall.
    # Require the torso to be tilted significantly more towards horizontal or body collapsed.
    if aspect < 0.75 and horiz < 0.48:
        return 0.0

    # For upright/standing postures that tilt heavily, scale smoothly from the 0.48 floor
    if aspect < 0.75:
        tilt_term = (horiz - 0.45) / 0.55
        return float(min(1.0, 0.60 * tilt_term))

    return float(0.6 * horiz + 0.4 * aspect_term)


def analyze(persons: List[PersonDetection], motion_score: float) -> List[Candidate]:
    """Return anomaly candidates from the current frame's poses + scene motion."""
    out: List[Candidate] = []

    # --- Fall / medical (per person) ---
    best_fall, fall_person = 0.0, None
    for p in persons:
        s = fall_score(p)
        if s > best_fall:
            best_fall, fall_person = s, p
    if best_fall >= 0.55 and fall_person is not None:
        # A collapse with little scene motion => likely medical (person down, not moving).
        reason = "person in a horizontal/collapsed posture"
        out.append((AnomalyType.FALL, float(min(1.0, best_fall)), reason))

    # --- Fight / violence (needs >=2 people close + motion) ---
    if len(persons) >= 2:
        best_prox = 0.0
        best_iou = 0.0
        aggro = 0.0
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                best_prox = max(best_prox, geometry.proximity(persons[i], persons[j]))
                best_iou = max(best_iou, geometry.box_iou(persons[i].box, persons[j].box))
            aggro = max(aggro, geometry.aggressive_pose(persons[i]))
        closeness = max(best_prox, best_iou)
        violence = closeness * (0.35 + 0.65 * motion_score) + 0.15 * aggro
        violence = float(np.clip(violence, 0.0, 1.0))
        if violence >= 0.45:
            out.append((
                AnomalyType.VIOLENCE, violence,
                "two or more people in close contact with rapid motion",
            ))

    return out
