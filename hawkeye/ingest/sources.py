"""Video source abstraction. cv2 is imported lazily so this module stays light."""
from __future__ import annotations

import logging
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

    On Wayland (default on modern Linux), uses `grim` to capture compositor
    pixels directly without Wayland security black-screen blocking.
    Falls back to `mss` on X11 / Windows / macOS.
    """

    def __init__(self, spec: str = "screen", width: int = 0, height: int = 0):
        super().__init__(spec, width, height, loop=False)
        self._sct = None
        self._mon = None
        self._is_wayland = False
        self._grim_cmd = None

    def open(self) -> "ScreenSource":
        import os
        import shutil

        if os.environ.get("WAYLAND_DISPLAY") and shutil.which("grim"):
            self._is_wayland = True
            cmd = ["grim", "-s", "1"]
            sub = self.spec.split(":", 1)[1].strip() if ":" in self.spec else ""
            if sub == "right":
                cmd.extend(["-g", "768,0 768x864"])
            elif sub == "left":
                cmd.extend(["-g", "0,0 768x864"])
            cmd.extend(["-t", "jpeg", "-q", "70", "-"])
            self._grim_cmd = cmd
            log.info("Opened Wayland screen capture using grim (%s)", " ".join(cmd))
            return self

        import mss

        self._sct = mss.MSS()
        monitor_idx = 1
        if ":" in self.spec:
            try:
                monitor_idx = int(self.spec.split(":", 1)[1])
            except ValueError:
                monitor_idx = 1
        if monitor_idx >= len(self._sct.monitors):
            monitor_idx = 1
        self._mon = self._sct.monitors[monitor_idx]
        log.info("Opened X11 screen capture (monitor %d: %dx%d)",
                 monitor_idx, self._mon["width"], self._mon["height"])
        return self

    def read(self) -> Optional[np.ndarray]:
        import cv2

        if self._is_wayland:
            import subprocess

            p = subprocess.run(self._grim_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if not p.stdout:
                return None
            frame = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return None
        else:
            if self._sct is None:
                raise RuntimeError("Source not opened; call open() first.")
            sct_img = self._sct.grab(self._mon)
            frame = np.ascontiguousarray(np.array(sct_img)[:, :, :3])

        if self.width and self.height and (frame.shape[1] != self.width or frame.shape[0] != self.height):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

    def release(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None
        self._grim_cmd = None

    @property
    def is_open(self) -> bool:
        return self._grim_cmd is not None or self._sct is not None


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

