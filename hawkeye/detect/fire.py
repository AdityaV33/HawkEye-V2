"""Fire and smoke detector (Hybrid: YOLO model + chromatic verification + flicker gating).

Approach C Architecture:
- If `fire_model` is configured and present, a fine-tuned YOLO model proposes candidate regions.
- A fast chromatic verification gate checks candidate boxes for physical high-temperature
  color characteristics (in BGR, YCrCb, and HSV color spaces) and edge variance,
  filtering out video screens, flat orange posters, and red clothing.
- If `fire_model` is empty or missing, it gracefully runs the zero-weight chromatic &
  flicker fallback detector, ensuring fire detection works out of the box without crashing.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("hawkeye.detect.fire")

FIRE_KEYWORDS = ("fire", "flame", "smoke", "blaze")


@dataclass
class FireDetection:
    box: np.ndarray             # [x1, y1, x2, y2]
    conf: float
    label: str                  # "fire", "flame", "smoke"
    chromatic_score: float = 1.0


class FireDetector:
    def __init__(
        self,
        model_path: str = "",
        device: str = "auto",
        conf: float = 0.40,
        keywords: Sequence[str] = FIRE_KEYWORDS,
        verify_chromatic: bool = True,
        chromatic_fallback: bool = True,
    ):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.keywords = tuple(k.lower() for k in keywords if k and k.strip())
        self.verify_chromatic = verify_chromatic
        self.chromatic_fallback = chromatic_fallback
        self._model = None
        self._prev_frame: Optional[np.ndarray] = None

        # Check if the model file physically exists
        self.has_model = bool(model_path) and Path(model_path).is_file()
        self.enabled = self.has_model or self.chromatic_fallback

        if model_path and not self.has_model:
            log.info(
                "fire_model %r is configured but file is not present — "
                "operating in chromatic & flicker verification fallback mode.",
                model_path,
            )
        elif not model_path and self.chromatic_fallback:
            log.info("No fire_model configured — using zero-weight chromatic & flicker detector.")

    MIN_COMPOUND_PREFIX = 3

    def _matches(self, label: str) -> bool:
        """Token-aware keyword match against a model class name."""
        tokens = [t for t in re.split(r"[^a-z0-9]+", label.lower()) if t]
        for t in tokens:
            for k in self.keywords:
                if t == k:
                    return True
                if t.endswith(k) and len(t) - len(k) >= self.MIN_COMPOUND_PREFIX:
                    return True
        return False

    def load(self) -> "FireDetector":
        if not self.has_model:
            return self

        try:
            from ultralytics import YOLO
            from .pose import resolve_device

            self.device = resolve_device(self.device)
            log.info("Loading fire model %s on %s", self.model_path, self.device)
            self._model = YOLO(self.model_path)
            self._model.to(self.device)
        except Exception as e:
            log.warning(
                "Failed to load fire model %s (%s) — falling back to chromatic detector.",
                self.model_path, e,
            )
            self.has_model = False
            self._model = None
        return self

    # --- Chromatic & Physical Verification ---

    def verify_flame_chromatics(self, roi: np.ndarray) -> Tuple[bool, float]:
        """Verify physical flame color properties in BGR, YCrCb, and HSV spaces.

        Celik & Demirel / Chen flame conditions:
        1. RGB space: R > G > B and R >= 140 (high red luminance).
        2. YCrCb space: Y >= Cb and Cr >= Cb + 10 (high chrominance red).
        3. HSV space: Hue in [0, 35] or [165, 180] (red-orange-yellow), S >= 40, V >= 70.
        4. Texture & Variance: Real flames possess turbulent luminance gradients (std >= 10).
        """
        import cv2

        if roi is None or roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
            return False, 0.0

        b = roi[:, :, 0].astype(np.float32)
        g = roi[:, :, 1].astype(np.float32)
        r = roi[:, :, 2].astype(np.float32)

        # 1. Flame color rule: red/orange dominance OR white-hot combustion core
        red_dominant = ((r >= g) & (g > b) & (r >= 130.0)) | ((r >= 210.0) & (g >= 190.0) & (b <= 210.0))

        # 2. YCrCb rule
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:, :, 0].astype(np.float32)
        Cr = ycrcb[:, :, 1].astype(np.float32)
        Cb = ycrcb[:, :, 2].astype(np.float32)
        ycrcb_rule = (Y >= Cb) & (Cr >= Cb + 8.0)

        # 3. HSV rule
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]
        hsv_rule = (((H <= 35) | (H >= 165)) & (S >= 35) & (V >= 65)) | ((S <= 50) & (V >= 210))

        # Flame mask
        flame_mask = red_dominant & (ycrcb_rule | (V >= 210)) & hsv_rule
        flame_pixels = int(np.count_nonzero(flame_mask))
        total_pixels = int(roi.shape[0] * roi.shape[1])
        density = flame_pixels / max(1, total_pixels)

        # Candidate must have at least 5% flame-colored pixels
        if density < 0.05:
            return False, density

        # 4. Variance check: reject completely flat solid colors (e.g. solid orange paper/signs)
        if flame_pixels >= 10:
            y_vals = Y[flame_mask]
            std_dev = float(np.std(y_vals))
            if std_dev < 8.0:
                return False, density

        chromatic_score = float(np.clip(density * 2.0, 0.1, 1.0))
        return True, chromatic_score

    def verify_smoke_chromatics(self, roi: np.ndarray) -> Tuple[bool, float]:
        """Verify smoke properties: low saturation, moderate luminance."""
        import cv2

        if roi is None or roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
            return False, 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

        S = hsv[:, :, 1]
        Y = ycrcb[:, :, 0]

        # Smoke is desaturated (S <= 75) across dark, gray, or white smoke (Y in [25, 235])
        smoke_mask = (S <= 75) & (Y >= 25) & (Y <= 235)
        smoke_pixels = int(np.count_nonzero(smoke_mask))
        total_pixels = int(roi.shape[0] * roi.shape[1])
        density = smoke_pixels / max(1, total_pixels)

        if density < 0.10:
            return False, density

        return True, float(np.clip(density, 0.1, 1.0))

    # --- Heuristic Fallback Detector ---

    def detect_heuristic(self, frame: np.ndarray) -> List[FireDetection]:
        """Zero-weight chromatic & temporal edge flicker detector."""
        import cv2

        h, w = frame.shape[:2]
        if h < 20 or w < 20:
            return []

        # Downscale for high-speed analysis
        scale = 320.0 / max(1, w)
        sw, sh = 320, max(1, int(h * scale))
        small = cv2.resize(frame, (sw, sh))

        b = small[:, :, 0].astype(np.float32)
        g = small[:, :, 1].astype(np.float32)
        r = small[:, :, 2].astype(np.float32)

        red_dominant = (r > g) & (g > b) & (r >= 145.0)

        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:, :, 0].astype(np.float32)
        Cr = ycrcb[:, :, 1].astype(np.float32)
        Cb = ycrcb[:, :, 2].astype(np.float32)
        ycrcb_rule = (Y >= Cb) & (Cr >= Cb + 12.0)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]
        hsv_rule = ((H <= 35) | (H >= 165)) & (S >= 45) & (V >= 85)

        mask = (red_dominant & ycrcb_rule & hsv_rule).astype(np.uint8) * 255

        # Morphological open to remove point noise, dilate to bridge flame tongues
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: List[FireDetection] = []
        frame_area = float(w * h)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Area in small frame: at least 20 pixels (~180 pixels in 960x540 frame)
            if area < 20:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)

            # Anti-card / anti-swatch geometric filter:
            # Flat UI cards, browser thumbnails, buttons, and solid swatches have near-perfect
            # rectangular fill (solidity >= 0.88) and straight orthogonal edges.
            # Real fire has chaotic, non-geometric contours with flame tongues and low solidity.
            rect_area = float(bw * bh)
            rectangularity = area / max(1.0, rect_area)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(1.0, hull_area)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

            # Reject orthogonal quadrilaterals / high-solidity cards
            if solidity >= 0.92 and rectangularity >= 0.80 and len(approx) <= 4:
                continue

            # Scale coordinates back to full resolution
            x1 = max(0, int(bx / scale))
            y1 = max(0, int(by / scale))
            x2 = min(w, int((bx + bw) / scale))
            y2 = min(h, int((by + bh) / scale))

            box_w = x2 - x1
            box_h = y2 - y1
            box_area = float(box_w * box_h)

            # Reject huge bounding boxes taking > 35% of frame (e.g. camera glare / orange room walls)
            if frame_area > 0 and (box_area / frame_area) > 0.35:
                continue

            roi = frame[y1:y2, x1:x2]
            valid_chromatic, c_score = self.verify_flame_chromatics(roi)
            if not valid_chromatic:
                continue

            # Temporal flicker check: reject completely static orange items (e.g. orange cone/backpack)
            # if we have consecutive frames and the region has zero temporal difference.
            if self._prev_frame is not None and self._prev_frame.shape == frame.shape:
                prev_roi = self._prev_frame[y1:y2, x1:x2]
                diff = cv2.absdiff(roi, prev_roi)
                diff_mean = float(diff.mean())
                # If motion inside the candidate flame is essentially 0 across frames, it is static clutter
                if diff_mean < 0.6:
                    continue

            conf = float(np.clip(0.60 + 0.30 * c_score, 0.60, 0.95))
            box_arr = np.array([x1, y1, x2, y2], dtype=np.float32)
            out.append(FireDetection(box=box_arr, conf=conf, label="fire", chromatic_score=c_score))

        self._prev_frame = frame.copy()
        return out

    # --- Main detect API ---

    def detect(self, frame: np.ndarray, max_area_ratio: float = 0.35) -> List[FireDetection]:
        """Detect fire/smoke in frame using YOLO + chromatic verification (or chromatic fallback)."""
        if not self.enabled:
            return []

        if self.has_model:
            if self._model is None:
                self.load()

        # If model is loaded, run YOLO + Chromatic gating
        if self._model is not None:
            try:
                results = self._model.predict(frame, conf=self.conf, device=self.device, verbose=False)
                out: List[FireDetection] = []
                if not results or results[0].boxes is None:
                    return out

                r = results[0]
                names = r.names
                h, w = frame.shape[:2]
                frame_area = float(w * h)

                for box, cls, cf in zip(
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.cls.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                ):
                    label = str(names.get(int(cls), int(cls))).lower()
                    if not self._matches(label):
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    box_w, box_h = max(0, x2 - x1), max(0, y2 - y1)
                    box_area = float(box_w * box_h)

                    if frame_area > 0 and (box_area / frame_area) > max_area_ratio:
                        continue

                    roi = frame[y1:y2, x1:x2]
                    chromatic_score = 1.0

                    if self.verify_chromatic:
                        is_smoke = "smoke" in label
                        if is_smoke:
                            valid, c_score = self.verify_smoke_chromatics(roi)
                        else:
                            valid, c_score = self.verify_flame_chromatics(roi)

                        if not valid:
                            # Rejected by physical chromatic verification (e.g. TV screen, poster, clothes)
                            continue
                        chromatic_score = c_score

                    out.append(FireDetection(box=box, conf=float(cf), label=label, chromatic_score=chromatic_score))
                return out
            except Exception as e:
                log.warning("YOLO fire inference failed (%s) — trying fallback detector.", e)

        # Fallback to zero-weight chromatic & flicker detector
        if self.chromatic_fallback:
            return self.detect_heuristic(frame)

        return []

