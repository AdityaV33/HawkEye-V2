"""Video source abstraction. cv2 is imported lazily so this module stays light."""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

log = logging.getLogger("hawkeye.ingest")


class VideoSource:
    """Base class. Yields BGR frames (numpy HxWx3) or None at end-of-stream."""

    def __init__(self, spec: str, width: int = 0, height: int = 0, loop: bool = False):
        self.spec = spec
        self.width = width
        self.height = height
        self.loop = loop
        self._cap = None

    def open(self) -> "VideoSource":
        import cv2

        import platform

        arg: object = self.spec
        if self.spec.isdigit():
            arg = int(self.spec)
            if platform.system() == "Windows":
                self._cap = cv2.VideoCapture(arg, cv2.CAP_DSHOW)
                if not self._cap.isOpened():
                    self._cap = cv2.VideoCapture(arg)
            else:
                self._cap = cv2.VideoCapture(arg)
        else:
            self._cap = cv2.VideoCapture(arg)
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._cap.isOpened():
            hint = ""
            if self.spec.isdigit():
                import platform
                if platform.system() == "Darwin":
                    hint = (" On macOS, the terminal/app running this needs camera "
                            "access: System Settings -> Privacy & Security -> Camera "
                            "-> enable it for your terminal app, then restart the terminal.")
            raise RuntimeError(
                f"Could not open video source '{self.spec}'. "
                f"Check the camera index / URL / file path.{hint}"
            )
        log.info("Opened video source: %s", self.spec)
        return self

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            raise RuntimeError("Source not opened; call open() first.")
        ok, frame = self._cap.read()
        if not ok:
            if self.loop:
                import cv2

                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if ok:
                    return frame
            return None
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.release()


class ScreenSource(VideoSource):
    """Real-time desktop screen capture source.

    On Wayland (default on modern Linux), uses fast uncompressed `grim -t ppm`
    inside a background grabber thread, eliminating capture wait time from the
    inference pipeline.
    On Windows, uses DXCam (DirectX Desktop Duplication API) if available for
    hardware-accelerated 60+ FPS capture, falling back to a dedicated background
    MSS grabber thread with zero-copy buffer views and fast linear downsampling.
    On X11 / macOS, uses the dedicated background MSS grabber thread.
    """

    def __init__(self, spec: str = "screen", width: int = 0, height: int = 0):
        super().__init__(spec, width, height, loop=False)
        self._sct = None
        self._mon = None
        self._dxcam = None
        self._is_wayland = False
        self._grim_cmd = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._has_frame = threading.Event()

    def open(self) -> "ScreenSource":
        import os
        import platform
        import shutil

        monitor_idx = 1
        if ":" in self.spec:
            try:
                monitor_idx = int(self.spec.split(":", 1)[1])
            except ValueError:
                monitor_idx = 1

        if os.environ.get("WAYLAND_DISPLAY") and shutil.which("grim"):
            self._is_wayland = True
            cmd = ["grim", "-s", "1"]
            sub = self.spec.split(":", 1)[1].strip() if ":" in self.spec else ""
            if sub == "right":
                cmd.extend(["-g", "768,0 768x864"])
            elif sub == "left":
                cmd.extend(["-g", "0,0 768x864"])
            # PPM is an uncompressed pixmap dump — 3x-4x faster than CPU JPEG encoding in grim
            cmd.extend(["-t", "ppm", "-"])
            self._grim_cmd = cmd
            log.info("Opened Wayland screen capture using grim (%s)", " ".join(cmd))
            self._running = True
            self._thread = threading.Thread(target=self._capture_worker_grim, daemon=True, name="ScreenSourceGrabberGrim")
            self._thread.start()
            self._has_frame.wait(timeout=2.0)
            return self

        if platform.system() == "Windows":
            try:
                import dxcam
                cam = dxcam.create(device_idx=0, output_idx=max(0, monitor_idx - 1), output_color="BGR")
                if cam is not None:
                    cam.start(target_fps=60, video_mode=True)
                    self._dxcam = cam
                    self._running = True
                    log.info("Opened Windows hardware-accelerated screen capture via DXCam (monitor %d)", monitor_idx)
                    return self
            except Exception as e:
                log.info("DXCam not used (%s); falling back to optimized background MSS grabber", e)

        import mss

        self._sct = mss.MSS()
        if monitor_idx >= len(self._sct.monitors):
            monitor_idx = 1
        self._mon = self._sct.monitors[monitor_idx]
        log.info("Opened screen capture via threaded MSS (monitor %d: %dx%d)",
                 monitor_idx, self._mon["width"], self._mon["height"])
        self._running = True
        self._thread = threading.Thread(target=self._capture_worker_mss, daemon=True, name="ScreenSourceGrabberMSS")
        self._thread.start()
        self._has_frame.wait(timeout=2.0)
        return self

    def _capture_worker_grim(self) -> None:
        import subprocess
        import cv2

        while self._running:
            p = subprocess.run(self._grim_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if not p.stdout:
                continue
            frame = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if self.width and self.height and (frame.shape[1] != self.width or frame.shape[0] != self.height):
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            with self._lock:
                self._latest_frame = frame
            self._has_frame.set()

    def _capture_worker_mss(self) -> None:
        import cv2
        import time

        while self._running and self._sct is not None:
            try:
                sct_img = self._sct.grab(self._mon)
                # Zero-copy view of raw BGRA buffer
                raw_bgra = np.frombuffer(sct_img.raw, dtype=np.uint8).reshape((sct_img.height, sct_img.width, 4))
                if self.width and self.height and (sct_img.width != self.width or sct_img.height != self.height):
                    # Resize 4-channel raw buffer with fast SIMD linear interpolation
                    resized = cv2.resize(raw_bgra, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                    frame = cv2.cvtColor(resized, cv2.COLOR_BGRA2BGR)
                else:
                    frame = cv2.cvtColor(raw_bgra, cv2.COLOR_BGRA2BGR)

                with self._lock:
                    self._latest_frame = frame
                self._has_frame.set()
                # Yield 5ms to avoid saturating 100% of a core when capture is faster than downstream
                time.sleep(0.005)
            except Exception:
                time.sleep(0.02)

    def read(self) -> Optional[np.ndarray]:
        import cv2

        if self._dxcam is not None:
            frame = self._dxcam.get_latest_frame()
            if frame is None:
                with self._lock:
                    return self._latest_frame.copy() if self._latest_frame is not None else None
            if self.width and self.height and (frame.shape[1] != self.width or frame.shape[0] != self.height):
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            with self._lock:
                self._latest_frame = frame
            return frame.copy()

        if not self._has_frame.wait(timeout=1.0):
            return None
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def release(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._dxcam is not None:
            try:
                self._dxcam.stop()
            except Exception:
                pass
            self._dxcam = None
        if self._sct is not None:
            self._sct.close()
            self._sct = None
        self._grim_cmd = None
        with self._lock:
            self._latest_frame = None

    @property
    def is_open(self) -> bool:
        return (self._is_wayland and self._running) or (self._dxcam is not None) or (self._sct is not None and self._running)


class DroneSource(VideoSource):
    """Phase 2 placeholder — a Tello stream plugs in here (djitellopy).

    Implementation deferred to Phase 2. It only needs to override open()/read()
    to yield BGR frames; everything downstream is unchanged.
    """

    def open(self):  # pragma: no cover - Phase 2
        raise NotImplementedError(
            "DroneSource is Phase 2. See docs/architecture.md — it will wrap "
            "djitellopy's frame_read to yield BGR frames through this same interface."
        )


def open_source(spec: str, width: int = 0, height: int = 0) -> VideoSource:
    """Factory: pick a source from the spec.

    - "0", "1", ...        -> webcam index
    - "screen", "screen:1" -> real-time desktop screen capture
    - http(s)/rtsp URL     -> IP camera (e.g. phone IP-webcam app)
    - "drone:..."          -> Phase 2 Tello source (not yet implemented)
    - anything else        -> video file (looped, handy for demos on datasets)
    """
    s = spec.strip()
    if s == "screen" or s.startswith("screen:"):
        return ScreenSource(s, width, height).open()
    if s.startswith("drone:"):
        return DroneSource(s.split(":", 1)[1], width, height).open()
    is_stream = s.isdigit() or s.startswith(("http://", "https://", "rtsp://", "rtmp://"))
    return VideoSource(s, width, height, loop=not is_stream).open()

