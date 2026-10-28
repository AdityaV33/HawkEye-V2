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
from .pose import (L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, NOSE,
                   PersonDetection)

Candidate = Tuple[AnomalyType, float, str]


def is_human(p: PersonDetection) -> bool:
    """Verify that a detection has human anatomical structure (not background clutter)."""
    # Reject micro-boxes and tiny thumbnail cards
    if p.width < 16 or p.height < 28:
        return False

    # Count keypoints with confidence >= 0.20
    kpts_conf = sum(1 for kp in p.keypoints if kp[2] >= 0.20)
    if kpts_conf < 3:
        return False
    # Must have at least one upper-body point (head/shoulders/arms: indices 0-10)
    # and at least one lower-body point (hips/knees/ankles: indices 11-16)
    has_upper = any(p.kp(i, min_conf=0.20) is not None for i in range(11))
    has_lower = any(p.kp(i, min_conf=0.20) is not None for i in range(11, 17))
    if not (has_upper and (has_lower or kpts_conf >= 5)):
        return False

    # A pose model can hallucinate a person on a dog or other elongated
    # object. Require a human-like head/torso anchor and bilateral torso
    # structure when those keypoints are available.
    head = p.kp(NOSE, min_conf=0.20)
    shoulders = [p.kp(i, min_conf=0.20) for i in (L_SHOULDER, R_SHOULDER)]
    hips = [p.kp(i, min_conf=0.20) for i in (L_HIP, R_HIP)]
    shoulder_pair = [pt for pt in shoulders if pt is not None]
    hip_pair = [pt for pt in hips if pt is not None]
    if head is None and len(shoulder_pair) < 2:
        return False
    if len(shoulder_pair) == 0 or len(hip_pair) == 0:
        return False

    shoulder_y = float(np.mean([pt[1] for pt in shoulder_pair]))
    hip_y = float(np.mean([pt[1] for pt in hip_pair]))
    # For upright people, torso anchors have a clear top-to-bottom order.
    # Horizontal fallen people are exempt because their axis is rotated.
    if p.height >= p.width and (
        head is not None and head[1] > shoulder_y + 0.12 * p.height
        or shoulder_y > hip_y + 0.08 * p.height
    ):
        return False
    return True


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


def is_standing(p: PersonDetection) -> bool:
    """True if person has vertically supporting legs (standing on feet).
    
    A standing person whose torso is bent (e.g. tying shoes, leaning, stretching)
    is standing and has not fallen.
    """
    lh = p.kp(L_HIP, min_conf=0.20)
    rh = p.kp(R_HIP, min_conf=0.20)
    lk = p.kp(13, min_conf=0.20)
    rk = p.kp(14, min_conf=0.20)
    la = p.kp(15, min_conf=0.20)
    ra = p.kp(16, min_conf=0.20)

    hip_y = min([pt[1] for pt in (lh, rh) if pt is not None], default=None)
    knee_y = max([pt[1] for pt in (lk, rk) if pt is not None], default=None)
    ankle_y = max([pt[1] for pt in (la, ra) if pt is not None], default=None)

    aspect = p.width / (p.height + 1e-6)

    # If ankles and hips detected, check if legs extend vertically downward
    if hip_y is not None and ankle_y is not None:
        vert_leg_span = (ankle_y - hip_y) / (p.height + 1e-6)
        if vert_leg_span >= 0.35 and aspect < 0.85:
            return True

    # If knees and hips detected (e.g. lower legs occluded)
    if hip_y is not None and knee_y is not None:
        vert_thigh_span = (knee_y - hip_y) / (p.height + 1e-6)
        if vert_thigh_span >= 0.20 and aspect < 0.75:
            return True

    return False


def fall_score(p: PersonDetection) -> float:
    """Single-frame fall/lying likelihood in [0, 1].
    
    A fall requires the legs to be parallel to the floor (lying down / collapsed),
    not merely a bent torso.
    """
    if not is_human(p):
        return 0.0

    # A standing person on their feet (even with bent torso) is not a fall
    if is_standing(p):
        return 0.0

    leg_horiz = geometry.leg_horizontalness(p)
    aspect = p.width / (p.height + 1e-6)          # >1 => wider than tall (lying)
    aspect_term = np.clip((aspect - 0.75) / 1.1, 0.0, 1.0)

    # During the brief collapse transition, pose estimation often keeps one
    # leg partly vertical even though the person is no longer standing. Keep
    # the strict floor-parallel requirement for static scenes, while allowing
    # the temporal/action path to use this weaker geometry during motion.
    if leg_horiz is not None and leg_horiz < 0.35:
        return 0.0

    sh, hip = torso_points(p)
    torso_horiz = geometry.torso_horizontalness(sh, hip)

    # An upright torso (sitting upright or standing straight) is not a fall
    if torso_horiz is not None and torso_horiz < 0.30:
        return 0.0

    # If leg keypoints are available and parallel to the floor (>= 0.50)
    if leg_horiz is not None:
        t_horiz = torso_horiz if torso_horiz is not None else leg_horiz
        return float(min(1.0, 0.55 * leg_horiz + 0.30 * t_horiz + 0.15 * aspect_term))

    # A cropped portrait, UI thumbnail, or seated upper body has no leg keypoints
    # and can be wide (e.g. 16:9 video thumbnails). Never blind-default horizontalness
    # or infer a fall without verified horizontal torso orientation.
    if torso_horiz is not None and torso_horiz >= 0.50 and aspect >= 1.35:
        return float(min(1.0, 0.50 * torso_horiz + 0.50 * aspect_term))

    # Torso alone without horizontal orientation or horizontal legs is NOT a fall
    return 0.0


def analyze(persons: List[PersonDetection], motion_score: float,
            violence_threshold: float = 0.65,
            violence_motion_min: float = 0.18) -> List[Candidate]:
    """Return anomaly candidates from the current frame's poses + scene motion."""
    out: List[Candidate] = []

    # --- Fall / medical (per person) ---
    best_fall, fall_person = 0.0, None
    for p in persons:
        s = fall_score(p)
        if s > best_fall:
            best_fall, fall_person = s, p
    # A moving transition can score below the static lying threshold because
    # the keypoint skeleton is still rotating. It is accepted only with
    # meaningful scene motion and still requires the non-standing geometry
    # checks in fall_score().
    if best_fall >= 0.52 or (best_fall >= 0.39 and motion_score >= 0.15):
        # A collapse with little scene motion => likely medical (person down, not moving).
        reason = "person in a horizontal/collapsed posture"
        out.append((AnomalyType.FALL, float(min(1.0, best_fall)), reason))

    # --- Fight / violence (needs >=2 people close + motion) ---
    if len(persons) >= 2:
        best_violence = 0.0
        for i in range(len(persons)):
            p1 = persons[i]
            for j in range(i + 1, len(persons)):
                p2 = persons[j]
                iou = geometry.box_iou(p1.box, p2.box)
                dist = float(np.linalg.norm(p1.centroid - p2.centroid))
                scale = (p1.height + p2.height) / 2.0 + 1e-6

                # Physical reach constraint: people in separate thumbnail cards or distant cannot be fighting
                if iou < 0.08 and dist > 0.65 * scale:
                    continue

                prox = geometry.proximity(p1, p2)
                closeness = max(iou, prox)
                aggro_1 = geometry.aggressive_pose(p1)
                aggro_2 = geometry.aggressive_pose(p2)
                pair_aggro = max(aggro_1, aggro_2)

                # Calm people standing together without mutual aggressive stance or box overlap are not fighting
                if pair_aggro < 0.35 and iou < 0.12:
                    continue

                pair_viol = closeness * (0.35 + 0.65 * motion_score) + 0.20 * pair_aggro
                pair_viol = float(np.clip(pair_viol, 0.0, 1.0))

                # Precision guard: close people in a thumbnail or calm conversation
                # can have high proximity but are not a police-worthy assault.
                if pair_viol >= violence_threshold and (
                        motion_score >= violence_motion_min or pair_aggro >= 0.5):
                    if pair_viol > best_violence:
                        best_violence = pair_viol

        if best_violence >= violence_threshold:
            out.append((
                AnomalyType.VIOLENCE, best_violence,
                "two or more people in close contact with rapid motion",
            ))

    return out
