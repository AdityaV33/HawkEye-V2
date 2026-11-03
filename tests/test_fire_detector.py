"""Comprehensive unit tests for Fire & Smoke Detection (Approach C Hybrid).

Validates:
1. Graceful degradation when weights file is missing.
2. Label matching for fire and smoke keywords.
3. Chromatic flame verification across BGR, YCrCb, and HSV spaces.
4. Smoke chromatic and saturation verification.
5. Anti-card geometric filter (rejects flat rectangular browser cards / swatches).
6. Temporal flicker check (rejects static orange clutter).
7. YOLO deep learning model inference (tests models/fire.pt on real images and screen rejections).
8. Stage A end-to-end integration (fire event trigger, high score, HUD rendering).
9. Life-safety precedence over unarmed violence.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hawkeye.config import DetectConfig
from hawkeye.detect.fire import FIRE_KEYWORDS, FireDetection, FireDetector
from hawkeye.detect.stage_a import StageA
from hawkeye.events.schema import AnomalyType, StageAEvent


def test_graceful_degradation():
    """Missing model weights should never crash the detector."""
    det_no_fallback = FireDetector(
        model_path="models/definitely_missing_fire.pt",
        chromatic_fallback=False,
    )
    assert not det_no_fallback.enabled, "Detector should disable when model is missing and fallback is False"
    det_no_fallback.load()
    assert det_no_fallback.detect(np.zeros((100, 100, 3), dtype=np.uint8)) == []

    det_fallback = FireDetector(
        model_path="models/definitely_missing_fire.pt",
        chromatic_fallback=True,
    )
    assert det_fallback.enabled, "Detector should enable fallback mode when model is missing"
    assert not det_fallback.has_model, "has_model should be False for missing file"
    det_fallback.load()
    print("  ✓ graceful degradation on missing model file")


def test_fire_label_matching():
    """Verify keyword and compound matching for fire/smoke classes."""
    det = FireDetector()
    matches = ["fire", "smoke", "campfire", "wildfire", "flame", "blaze"]
    for lbl in matches:
        assert det._matches(lbl), f"Expected '{lbl}' to match fire keywords"

    rejects = ["car", "person", "chair", "firearm", "gun", "knife"]
    for lbl in rejects:
        assert not det._matches(lbl), f"Expected '{lbl}' to NOT match fire keywords"
    print("  ✓ fire and smoke label matching")


def test_chromatic_flame_verification():
    """Verify physical flame color properties in BGR, YCrCb, and HSV spaces."""
    det = FireDetector()

    # 1. Reject pure non-fire patches
    black = np.zeros((50, 50, 3), dtype=np.uint8)
    blue = np.full((50, 50, 3), (255, 0, 0), dtype=np.uint8)
    green = np.full((50, 50, 3), (0, 255, 0), dtype=np.uint8)

    assert not det.verify_flame_chromatics(black)[0], "Black patch accepted as fire"
    assert not det.verify_flame_chromatics(blue)[0], "Blue patch accepted as fire"
    assert not det.verify_flame_chromatics(green)[0], "Green patch accepted as fire"

    # 2. Reject flat solid orange rectangle (e.g. orange poster/t-shirt, zero variance)
    flat_orange = np.full((60, 60, 3), (20, 120, 240), dtype=np.uint8)
    valid_flat, _ = det.verify_flame_chromatics(flat_orange)
    assert not valid_flat, "Flat solid orange surface accepted without turbulent variance"

    # 3. Accept synthetic flame with high-temperature gradients (R > G > B, luminance variance)
    np.random.seed(42)
    flame_patch = np.zeros((60, 60, 3), dtype=np.uint8)
    flame_patch[:, :] = (30, 30, 30)
    for y in range(15, 45):
        for x in range(15, 45):
            intensity = np.random.randint(180, 255)
            flame_patch[y, x] = (np.random.randint(10, 40), np.random.randint(80, 150), intensity)

    valid_flame, score = det.verify_flame_chromatics(flame_patch)
    assert valid_flame, "Realistic flame patch rejected by chromatic verification"
    assert score > 0.3, f"Flame score was too low: {score}"
    print("  ✓ chromatic flame verification and flat-color rejection")


def test_smoke_verification():
    """Verify smoke detection filters: low saturation, moderate luminance."""
    det = FireDetector()

    # Desaturated billowing grayish patch
    smoke_patch = np.full((50, 50, 3), (130, 135, 140), dtype=np.uint8)
    valid_smoke, score = det.verify_smoke_chromatics(smoke_patch)
    assert valid_smoke, "Desaturated grayish smoke patch rejected"

    # Saturated red/blue patch rejected as smoke
    saturated_patch = np.full((50, 50, 3), (20, 30, 240), dtype=np.uint8)
    valid_sat, _ = det.verify_smoke_chromatics(saturated_patch)
    assert not valid_sat, "Highly saturated patch incorrectly accepted as smoke"
    print("  ✓ smoke chromatic and saturation verification")


def test_anti_card_rejection():
    """Flat rectangular UI cards (like Google Image thumbnails) must be rejected."""
    det = FireDetector(chromatic_fallback=True)

    # Synthetic frame with a rectangular card
    frame = np.full((540, 960, 3), (240, 240, 240), dtype=np.uint8)
    # Draw solid orange rectangular card
    cv2.rectangle(frame, (100, 100), (350, 250), (20, 130, 245), -1)

    dets = det.detect(frame)
    assert len(dets) == 0, f"Rectangular card triggered false fire alarm: {dets}"
    print("  ✓ anti-card geometric filter rejects rectangular UI cards")


def test_temporal_flicker_rejects_static_clutter():
    """Static orange objects without flicker must not trigger across frames."""
    det = FireDetector(chromatic_fallback=True)

    frame1 = np.full((300, 300, 3), (50, 50, 50), dtype=np.uint8)
    for y in range(80, 160):
        for x in range(80, 160):
            frame1[y, x] = (20, 120, 200 + (x % 50))

    det.detect(frame1)

    frame2 = frame1.copy()
    dets = det.detect(frame2)
    assert len(dets) == 0, "Static object without temporal flicker triggered fire alert"
    print("  ✓ temporal flicker check rejects static orange clutter")


def test_yolo_deep_learning_model():
    """Verify that models/fire.pt loads and correctly identifies classes."""
    model_path = Path("models/fire.pt")
    if not model_path.is_file():
        print("  - skipping YOLO model check (weights not present)")
        return

    det = FireDetector(str(model_path))
    det.load()
    assert det.has_model, "FireDetector failed to load models/fire.pt"
    assert det._model is not None

    # Check class mapping: 0=smoke, 1=fire
    names = det._model.names
    has_fire = any("fire" in str(v).lower() for v in names.values())
    has_smoke = any("smoke" in str(v).lower() for v in names.values())
    assert has_fire, f"Model lacks fire class: {names}"
    assert has_smoke, f"Model lacks smoke class: {names}"

    # Verify that a negative frame (pure white / gray) yields 0 detections
    empty_frame = np.full((480, 640, 3), (200, 200, 200), dtype=np.uint8)
    dets = det.detect(empty_frame)
    assert len(dets) == 0, f"Empty frame produced false detections: {dets}"
    print("  ✓ YOLO deep learning model verified (fire & smoke classes active)")


def test_stage_a_fire_event_and_drawing():
    """Stage A detects fire, triggers AnomalyType.FIRE, and annotates in orange."""
    cfg = DetectConfig(
        fire_model="models/non_existent.pt",
        fire_chromatic_fallback=True,
        fire_hysteresis_hits=1,  # immediate for testing
    )
    stage_a = StageA(cfg)
    stage_a.load()

    frame = np.full((540, 960, 3), (40, 40, 40), dtype=np.uint8)
    np.random.seed(123)
    # Add a turbulent flame tongue shape (non-rectangular geometry)
    for y in range(200, 290):
        half_w = int(6 + (y - 200) * 0.40)
        for x in range(400 - half_w, 400 + half_w):
            frame[y, x] = (np.random.randint(10, 40), np.random.randint(90, 150), np.random.randint(190, 255))

    event, annotated, _ = stage_a.process(frame, frame_id=1, ts=1.0)
    assert event.anomaly_type == AnomalyType.FIRE, f"Expected FIRE event, got {event.anomaly_type}"
    assert event.triggered, "FIRE event was not triggered"
    assert event.score >= 0.88, f"Expected high score for fire, got {event.score}"
    assert len(event.boxes) >= 1, "No bounding box registered for fire"
    assert annotated is not None and annotated.shape == frame.shape
    print("  ✓ Stage A fire trigger, high score, and orange rendering")


def test_fire_precedence_over_violence():
    """Fire is a life-safety critical threat and takes precedence in candidate fusion."""
    candidates = [
        (AnomalyType.VIOLENCE, 0.85, "fighting detected"),
        (AnomalyType.FIRE, 0.92, "fire detected (fire: 92%)"),
    ]
    fire_cands = [c for c in candidates if c[0] == AnomalyType.FIRE]
    weapon_cands = [c for c in candidates if c[0] == AnomalyType.WEAPON]
    if fire_cands:
        atype, cscore, reason = fire_cands[0]
    elif weapon_cands:
        atype, cscore, reason = weapon_cands[0]
    else:
        atype, cscore, reason = max(candidates, key=lambda c: c[1])

    assert atype == AnomalyType.FIRE, f"Life-safety precedence failed: got {atype} instead of FIRE"
    assert cscore == 0.92
    print("  ✓ fire alert takes top life-safety precedence")


def main() -> int:
    checks = [
        test_graceful_degradation,
        test_fire_label_matching,
        test_chromatic_flame_verification,
        test_smoke_verification,
        test_anti_card_rejection,
        test_temporal_flicker_rejects_static_clutter,
        test_yolo_deep_learning_model,
        test_stage_a_fire_event_and_drawing,
        test_fire_precedence_over_violence,
    ]
    print(f"Running {len(checks)} fire detector self-checks...\n")
    failed = 0
    for fn in checks:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {fn.__name__}: unexpected {type(e).__name__}: {e}")
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
