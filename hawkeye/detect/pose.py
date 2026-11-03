"""YOLO11 pose + person detection. Heavy imports (torch/ultralytics) are lazy."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

log = logging.getLogger("hawkeye.detect.pose")

# COCO-17 keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16

# Standard COCO-17 topology, grouped by body part so the debug overlay can
# color-code limbs instead of drawing one undifferentiated skeleton — makes a
# misdetected elbow/knee/eye actually visible instead of blending into the rest.
#
# Note on coverage: COCO-17 has no toe/heel keypoints — the ankle is the last
# lower-body point YOLO11-pose (or any standard COCO model) predicts. Real toe
# keypoints require a different keypoint format entirely (e.g. COCO-WholeBody,
# 133 points) and a model trained on it; that's a model swap, not a rendering
# fix. Nothing here can conjure foot detail a 17-point model was never given.
FACE_EDGES = [(NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR),
             (NOSE, L_SHOULDER), (NOSE, R_SHOULDER)]
ARM_EDGES = [(L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
            (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST)]
LEG_EDGES = [(L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
            (R_HIP, R_KNEE), (R_KNEE, R_ANKLE)]
# Shoulder line, hip line, and the two diagonal torso sides. The synthetic
# neck->hip-center spine (drawn separately in stage_a._draw, not a real
# keypoint pair) is what actually reads as "vertical" — these alone don't.
TORSO_EDGES = [(L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP),
              (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP)]

SKELETON = FACE_EDGES + ARM_EDGES + LEG_EDGES + TORSO_EDGES


def resolve_device(pref: str = "auto") -> str:
    """Resolve 'auto' to mps (Mac) / cuda (3050) / cpu."""
    if pref and pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - torch missing
        pass
    return "cpu"


@dataclass
class PersonDetection:
    box: np.ndarray            # [x1, y1, x2, y2]
    conf: float
    keypoints: np.ndarray      # (17, 3) -> x, y, kp_conf
    track_id: Optional[int] = field(default=None)

    @property
    def width(self) -> float:
        return float(self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])

    @property
    def centroid(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    def kp(self, idx: int, min_conf: float = 0.3) -> Optional[np.ndarray]:
        """Return (x, y) for a keypoint if confident enough, else None."""
        x, y, c = self.keypoints[idx]
        return np.array([x, y]) if c >= min_conf else None


def is_valid_human_detection(box: np.ndarray, kp: np.ndarray, min_kpt_conf: float = 0.20) -> bool:
    """Strictly verify that a bounding box and keypoints represent a genuine human being.
    
    Rejects inanimate clutter, furniture, background textures, and noisy detections.
    """
    bw = max(1.0, float(box[2] - box[0]))
    bh = max(1.0, float(box[3] - box[1]))
    aspect = bw / bh

    # Reject extreme sliver boxes (e.g. door jambs, wall edges, long horizontal rails)
    if aspect > 3.2 or aspect < 0.18:
        return False

    # Require minimum pixel size (reject tiny thumbnail icons, UI clutter, and noise)
    if bw < 16 or bh < 28:
        return False

    # Require at least 4 keypoints with confidence >= min_kpt_conf
    kpts_conf = kp[:, 2] >= min_kpt_conf
    valid_count = int(np.sum(kpts_conf))
    if valid_count < 4:
        return False

    # Keypoint vertical spread: keypoints must span at least 25% of the bbox height
    valid_ys = kp[kpts_conf, 1]
    if (float(np.max(valid_ys)) - float(np.min(valid_ys))) < 0.25 * bh:
        return False

    # Anatomical anchors: must have upper body point (head/shoulders/arms) AND (lower body point (hips/legs) OR >= 5 confident keypoints for seated/half-body)
    has_upper = bool(np.any(kpts_conf[0:11]))
    has_lower = bool(np.any(kpts_conf[11:17]))
    if not (has_upper and (has_lower or valid_count >= 5)):
        return False

    # For upright/standing humans (bh >= bw), head/shoulders must be higher (smaller y) than lower body
    if bh >= bw:
        upper_ys = kp[kpts_conf & np.array([i < 11 for i in range(17)]), 1]
        lower_ys = kp[kpts_conf & np.array([i >= 11 for i in range(17)]), 1]
        if len(upper_ys) > 0 and len(lower_ys) > 0:
            if float(np.mean(upper_ys)) > float(np.mean(lower_ys)) + 0.10 * bh:
                # Inverted anatomy: upper body is lower than hips/feet!
                return False

    return True


class PoseDetector:
    def __init__(self, model_path: str = "yolo11n-pose.pt", device: str = "auto",
                 conf: float = 0.35):
        self.model_path = model_path
        self.device = resolve_device(device)
        self.conf = conf
        # Run YOLO inference with a calibrated floor (0.28) to catch distant people,
        # then strictly verify human anatomical structure on all candidates.
        self.predict_conf = min(0.28, conf)
        self._model = None

    def load(self) -> "PoseDetector":
        from ultralytics import YOLO

        log.info("Loading pose model %s on %s", self.model_path, self.device)
        self._model = YOLO(self.model_path)
        try:
            self._model.to(self.device)
        except Exception as e:
            log.warning("Could not move pose model to %s: %s", self.device, e)
        return self

    def _parse(self, r) -> List[PersonDetection]:
        """Turn one ultralytics Result into PersonDetections, strictly filtering for genuine humans."""
        out: List[PersonDetection] = []
        if r is None or r.keypoints is None or r.boxes is None:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kpts = r.keypoints.data.cpu().numpy()   # (N, 17, 3)
        for i in range(len(boxes)):
            # YOLO pose checkpoints may still return non-person detections
            # when fed broad/custom inputs. COCO class 0 is person; reject
            # every other class before anatomical heuristics run.
            if int(classes[i]) != 0:
                continue
            box = boxes[i]
            cf = float(confs[i])
            kp = kpts[i]

            # Require strict human anatomical structure on EVERY detection
            # so random clutter, furniture, background textures are NEVER classified as human
            if is_valid_human_detection(box, kp, min_kpt_conf=0.20):
                out.append(PersonDetection(box=box, conf=cf, keypoints=kp))

        return out

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        if self._model is None:
            self.load()
        results = self._model.predict(
            frame, conf=self.predict_conf, device=self.device, verbose=False
        )
        return self._parse(results[0]) if results else []

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[PersonDetection]]:
        """Pose for several frames in ONE forward pass. Returns one list per frame.

        Live inference is inherently one-frame-at-a-time, but offline feature
        extraction is not, and a per-frame ``predict()`` spends most of its wall
        clock on Python + launch overhead with the GPU idle between frames — which
        is why extracting a training set takes hours. Ultralytics batches a list
        natively, so this is the same model at the same ``conf`` producing the same
        detections; only the number of forward passes changes.

        Callers should batch frames **from one video**: identical frame sizes mean
        identical letterboxing, so batched results match the per-frame ones.
        """
        if not frames:
            return []
        if self._model is None:
            self.load()
        results = self._model.predict(
            frames, conf=self.predict_conf, device=self.device, verbose=False
        )
        out = [self._parse(r) for r in (results or [])]
        # predict() returns one Result per input; pad rather than lose alignment
        # with the caller's frame list if a backend ever returns fewer.
        out.extend([] for _ in range(len(frames) - len(out)))
        return out
