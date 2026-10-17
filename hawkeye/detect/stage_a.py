"""Stage A orchestrator: pose + heuristics + motion + weapons -> anomaly event.

Also renders the annotated frame shown on the dashboard.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from ..config import DetectConfig
from ..events.schema import AnomalyType, BBox, StageAEvent
from . import geometry, heuristics
from .action_classifier import ActionClassifier
from .pose import (ARM_EDGES, FACE_EDGES, L_ANKLE, L_HIP, L_SHOULDER,
                   L_WRIST, LEG_EDGES, R_ANKLE, R_HIP, R_SHOULDER, R_WRIST,
                   TORSO_EDGES, PersonDetection, PoseDetector)
from .scene_anomaly import MotionAnomaly
from .weapons import WEAPON_KEYWORDS, WeaponDetector

log = logging.getLogger("hawkeye.detect.stage_a")

# BGR colors
GREEN, RED, YELLOW, WHITE, CYAN, ORANGE = (80, 220, 80), (60, 60, 235), (40, 210, 235), (240, 240, 240), (235, 200, 60), (0, 140, 255)
# Per-limb-group debug colors (BGR), used only when the person isn't in alert
# state — an alert paints the whole skeleton red so it stays the dominant
# signal. Chosen to be distinguishable from each other and from RED/GREEN/CYAN.
FACE_CLR, ARM_CLR, LEG_CLR, TORSO_CLR = (235, 200, 60), (60, 190, 255), (255, 140, 60), (240, 240, 240)
SPINE_CLR = (0, 220, 255)  # bright amber — the vertical reference line



class StageA:
    def __init__(self, cfg: DetectConfig):
        self.cfg = cfg
        self.pose = PoseDetector(cfg.pose_model, cfg.device, cfg.person_conf)
        self.motion = MotionAnomaly()
        self.weapons = WeaponDetector(cfg.weapon_model, cfg.device, cfg.weapon_conf,
                                      cfg.weapon_labels or WEAPON_KEYWORDS)
        self.action = ActionClassifier(cfg.action_model, cfg.device,
                                       cfg.action_min_conf, cfg.action_stride,
                                       cfg.violence_motion_min,
                                       cfg.action_hysteresis_window,
                                       cfg.action_hysteresis_hits)
        self._last_trigger: Dict[str, float] = {}
        self._weapon_streak: int = 0

    def load(self) -> "StageA":
        # Surface the resolved compute device (auto -> cuda/mps/cpu) so a
        # "why is this slow on a GPU box" question is answerable from the logs.
        log.info("Stage A device: %s (requested '%s')", self.pose.device, self.cfg.device)
        self.pose.load()
        self.weapons.load()
        if self.action.available:
            self.action.load()
            log.info("Stage A action source: TRAINED model (%s)", self.cfg.action_model)
        else:
            log.info("Stage A action source: heuristics (no trained model configured)")
        return self

    def _filter_weapons(self, weapon_dets: List[WeaponDetection],
                        persons: List[PersonDetection]) -> List[WeaponDetection]:
        """Spatial gating: a brandished weapon must be actively held by a verified person.

        Strictly enforces that the weapon is in or immediately touching a human hand/wrist.
        Rejects background clutter, shoes/socks on feet, and chest/t-shirt designs.
        """
        if not weapon_dets:
            return []
        if not self.cfg.weapon_require_person:
            return weapon_dets
        if not persons:
            return []

        valid: List[WeaponDetection] = []
        for wd in weapon_dets:
            wx1, wy1, wx2, wy2 = wd.box
            wcx, wcy = (wx1 + wx2) / 2.0, (wy1 + wy2) / 2.0
            w_w = max(1.0, wx2 - wx1)
            w_h = max(1.0, wy2 - wy1)
            w_area = w_w * w_h

            held_by_someone = False
            for p in persons:
                if not heuristics.is_human(p):
                    continue

                px1, py1, px2, py2 = p.box
                pw = max(1.0, px2 - px1)
                ph = max(1.0, py2 - py1)
                p_scale = max(pw, ph)

                # Size sanity: handheld weapon is between 0.2% and 30% of the person's area
                p_area = max(1.0, pw * ph)
                if (w_area / p_area) > 0.30 or (w_area / p_area) < 0.002:
                    continue

                def dist_to_box(pt: np.ndarray) -> float:
                    dx = max(0.0, max(wx1 - pt[0], pt[0] - wx2))
                    dy = max(0.0, max(wy1 - pt[1], pt[1] - wy2))
                    return float(np.hypot(dx, dy))

                # Check wrist proximity (must be held in or immediately adjacent to hand)
                wrist_pts = [p.kp(idx, min_conf=0.20) for idx in (L_WRIST, R_WRIST)]
                wrist_dists = [dist_to_box(pt) for pt in wrist_pts if pt is not None]

                if not wrist_dists:
                    continue

                min_wrist_dist = min(wrist_dists)
                grasp_limit = 0.14 * p_scale   # Hand grasp reach radius

                if min_wrist_dist > grasp_limit:
                    continue

                # Exclude feet / shoes (reject white socks/shoes detected as knife/gun)
                ankle_pts = [p.kp(idx, min_conf=0.20) for idx in (L_ANKLE, R_ANKLE)]
                ankle_dists = [dist_to_box(pt) for pt in ankle_pts if pt is not None]
                if ankle_dists and min(ankle_dists) < min_wrist_dist:
                    continue

                # Exclude chest / t-shirt graphic: if wrist is not inside the box,
                # check if weapon is centered on the sternum between shoulders and hips
                wrist_inside = any(wx1 <= pt[0] <= wx2 and wy1 <= pt[1] <= wy2
                                   for pt in wrist_pts if pt is not None)
                if not wrist_inside:
                    ls = p.kp(L_SHOULDER, min_conf=0.20)
                    rs = p.kp(R_SHOULDER, min_conf=0.20)
                    lh = p.kp(L_HIP, min_conf=0.20)
                    rh = p.kp(R_HIP, min_conf=0.20)
                    if (ls is not None or rs is not None) and (lh is not None or rh is not None):
                        sh_y = min(pt[1] for pt in (ls, rs) if pt is not None)
                        hip_y = max(pt[1] for pt in (lh, rh) if pt is not None)
                        chest_cx = (px1 + px2) / 2.0
                        chest_cy = (sh_y + hip_y) / 2.0
                        d_chest = float(np.hypot(wcx - chest_cx, wcy - chest_cy))
                        if d_chest < min_wrist_dist:
                            continue

                held_by_someone = True
                break

            if held_by_someone:
                valid.append(wd)
        return valid

    # --- core ---
    def process(self, frame: np.ndarray, frame_id: int, ts: float
                ) -> Tuple[StageAEvent, np.ndarray, List[PersonDetection]]:
        persons = self.pose.detect(frame)
        motion = self.motion.update(frame)
        h, w = frame.shape[:2]

        # Trained action classifier takes precedence over the bootstrap
        # heuristics once weights are configured (see docs/training.md).
        if self.action.available:
            candidates = self.action.update(persons, motion, w, h)
        else:
            candidates = heuristics.analyze(persons, motion)

        # Detect weapons with oversized-box rejection (cars, ceiling fans, walls)
        raw_weapons = self.weapons.detect(frame, max_area_ratio=self.cfg.weapon_max_area_ratio)
        weapon_dets = self._filter_weapons(raw_weapons, persons)

        # Temporal hysteresis: require weapon across consecutive frames to reject 1-frame blips
        if weapon_dets:
            self._weapon_streak += 1
        else:
            self._weapon_streak = 0

        if weapon_dets and self._weapon_streak >= self.cfg.weapon_hysteresis_hits:
            top = max(wd.conf for wd in weapon_dets)
            candidates.append((AnomalyType.WEAPON, float(top),
                               f"weapon detected ({weapon_dets[0].label}: {top:.0%})"))

        # Open-vocabulary safety net: unusual motion with people present but no
        # specific type -> let Stage B reason about "any other bad thing".
        if motion >= 0.6 and persons:
            candidates.append((AnomalyType.UNKNOWN, float(self.cfg.motion_weight + 0.5 * motion),
                               "unusual scene motion"))

        # Fuse: highest-scoring candidate wins; motion floor keeps a baseline.
        if candidates:
            atype, cscore, reason = max(candidates, key=lambda c: c[1])
        else:
            atype, cscore, reason = AnomalyType.NORMAL, 0.0, ""
        score = float(np.clip(max(cscore, self.cfg.motion_weight * motion), 0.0, 1.0))

        triggered = self._maybe_trigger(atype, score, ts)

        event = StageAEvent(
            frame_id=frame_id, ts=ts, anomaly_type=atype, score=score,
            motion_score=float(motion), person_count=len(persons),
            boxes=[BBox(x1=float(p.box[0]), y1=float(p.box[1]),
                        x2=float(p.box[2]), y2=float(p.box[3])) for p in persons],
            triggered=triggered, reason=reason,
        )
        annotated = self._draw(frame, persons, weapon_dets, event)
        return event, annotated, persons

    def _maybe_trigger(self, atype: AnomalyType, score: float, ts: float) -> bool:
        if atype == AnomalyType.NORMAL or score < self.cfg.trigger_threshold:
            return False
        last = self._last_trigger.get(atype.value, -9999.0)
        if ts - last < self.cfg.cooldown_seconds:
            return False
        self._last_trigger[atype.value] = ts
        return True

    # --- rendering ---
    def _draw(self, frame, persons, weapon_dets, event: StageAEvent) -> np.ndarray:
        import cv2

        img = frame.copy()
        alert = event.triggered

        def line(a, b, color, thickness=2):
            if a is not None and b is not None:
                cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, thickness)

        best_fall_p = max(persons, key=heuristics.fall_score) if persons else None
        scene_fall = (event.anomaly_type == AnomalyType.FALL and (alert or event.score >= 0.35))

        for p in persons:
            x1, y1, x2, y2 = map(int, p.box)
            f_score = heuristics.fall_score(p)
            is_fall = (f_score >= 0.38) or (scene_fall and p is best_fall_p and f_score >= 0.25)

            if is_fall:
                person_color = ORANGE
                pct = int(max(f_score, event.score if scene_fall else 0.0) * 100)
                person_label = f"FALL {pct}%" if pct > 0 else "FALLING"
            elif alert:
                person_color = RED
                person_label = event.anomaly_type.value.upper()
            else:
                person_color = GREEN
                person_label = None

            # Bounding box (thicker for alert/fall)
            box_thick = 3 if (alert or is_fall) else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), person_color, box_thick)

            # Badge tag above head (or inside top if near image boundary)
            if person_label:
                (tw, th), _ = cv2.getTextSize(person_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                if y1 - th - 10 >= 0:
                    by1, by2 = y1 - th - 8, y1
                    ty = y1 - 4
                else:
                    by1, by2 = y1, y1 + th + 8
                    ty = y1 + th + 4
                cv2.rectangle(img, (x1, by1), (x1 + tw + 8, by2), person_color, -1)
                cv2.putText(img, person_label, (x1 + 4, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            if is_fall:
                # Falling state: bright orange skeleton
                for a, b in FACE_EDGES + ARM_EDGES + LEG_EDGES + TORSO_EDGES:
                    line(p.kp(a), p.kp(b), ORANGE, 3)
            elif alert:
                # Other alert state: uniform red skeleton
                for a, b in FACE_EDGES + ARM_EDGES + LEG_EDGES + TORSO_EDGES:
                    line(p.kp(a), p.kp(b), RED, 2)
            else:
                # Normal state: color-coded by body part
                for a, b in FACE_EDGES:
                    line(p.kp(a), p.kp(b), FACE_CLR)
                for a, b in ARM_EDGES:
                    line(p.kp(a), p.kp(b), ARM_CLR)
                for a, b in LEG_EDGES:
                    line(p.kp(a), p.kp(b), LEG_CLR)
                for a, b in TORSO_EDGES:
                    line(p.kp(a), p.kp(b), TORSO_CLR, 1)

            neck = geometry.mid(p.kp(L_SHOULDER), p.kp(R_SHOULDER))
            hip_c = geometry.mid(p.kp(L_HIP), p.kp(R_HIP))
            spine_color = ORANGE if is_fall else (RED if alert else SPINE_CLR)
            line(neck, hip_c, spine_color, 3)

            joint_color = (0, 200, 255) if is_fall else (RED if alert else YELLOW)
            for idx in range(17):
                x, y, c = p.keypoints[idx]
                if c >= 0.3:
                    r = 4 if idx >= 5 else 2
                    cv2.circle(img, (int(x), int(y)), r, joint_color, -1)

        for w in weapon_dets:
            x1, y1, x2, y2 = map(int, w.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), RED, 3)
            cv2.putText(img, f"WEAPON {w.conf:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

        self._draw_hud(img, event, cv2)
        return img

    def _draw_hud(self, img, event: StageAEvent, cv2) -> None:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 40), (25, 25, 25), -1)
        label = event.anomaly_type.value.upper()
        if event.anomaly_type == AnomalyType.FALL and (event.triggered or event.score >= 0.35):
            color = ORANGE
        elif event.triggered:
            color = RED
        else:
            color = WHITE if event.score > 0.2 else GREEN
        cv2.putText(img, f"{label}", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, f"people:{event.person_count}", (w - 150, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

        # score bar with trigger-threshold marker
        bx, by, bw, bh = 200, 12, 260, 16
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (70, 70, 70), -1)
        fill = int(bw * event.score)
        cv2.rectangle(img, (bx, by), (bx + fill, by + bh), color, -1)
        tx = bx + int(bw * self.cfg.trigger_threshold)
        cv2.line(img, (tx, by - 2), (tx, by + bh + 2), CYAN, 2)
        if event.triggered:
            cv2.putText(img, "! TRIGGER", (bx + bw + 12, by + bh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
