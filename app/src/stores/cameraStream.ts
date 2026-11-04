/**
 * Persistent camera streaming store.
 *
 * Keeps the phone camera active and broadcasts JPEG frames to the HawkEye
 * server (/ws/camera) over WebSocket. Designed to survive tab switches so the
 * user can leave the phone propped up in the Camera tab, tap Broadcast, and
 * freely view the Live tab while the phone continues to stream frames.
 */
import { Platform } from 'react-native';
import { create } from 'zustand';

type FacingMode = 'environment' | 'user';

type CameraStreamState = {
  isCameraActive: boolean;
  isBroadcasting: boolean;
  facingMode: FacingMode;
  fps: number;
  framesSent: number;
  error: string | null;

  // Actions
  startCamera: (facing?: FacingMode) => Promise<boolean>;
  stopCamera: () => void;
  toggleFacing: () => Promise<void>;
  startBroadcasting: (serverBase: string) => void;
  stopBroadcasting: () => void;
  attachVideoElement: (el: HTMLVideoElement | null) => void;
};

let activeMediaStream: MediaStream | null = null;
let broadcastWs: WebSocket | null = null;
let broadcastInterval: any = null;
let offscreenCanvas: HTMLCanvasElement | null = null;
let attachedVideoEl: HTMLVideoElement | null = null;
let lastFrameCount = 0;
let lastFpsTs = Date.now();
let framesInSecond = 0;

export const useCameraStream = create<CameraStreamState>((set, get) => ({
  isCameraActive: false,
  isBroadcasting: false,
  facingMode: 'environment',
  fps: 0,
  framesSent: 0,
  error: null,

  attachVideoElement: (el) => {
    attachedVideoEl = el;
    if (el && activeMediaStream) {
      el.srcObject = activeMediaStream;
      el.play().catch(() => {});
    }
  },

  startCamera: async (facing) => {
    if (Platform.OS !== 'web') {
      set({ error: 'Device camera streaming is currently supported in web browsers.' });
      return false;
    }

    const desiredFacing = facing ?? get().facingMode;
    if (activeMediaStream) {
      activeMediaStream.getTracks().forEach((t) => t.stop());
      activeMediaStream = null;
    }

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Camera access not supported by this browser.');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: desiredFacing },
          width: { ideal: 960 },
          height: { ideal: 540 },
        },
        audio: false,
      });

      activeMediaStream = stream;
      if (attachedVideoEl) {
        attachedVideoEl.srcObject = stream;
        await attachedVideoEl.play().catch(() => {});
      }

      set({
        isCameraActive: true,
        facingMode: desiredFacing,
        error: null,
      });
      return true;
    } catch (err: any) {
      const msg = err?.message || 'Could not access device camera. Check browser permissions.';
      set({ isCameraActive: false, error: msg });
      return false;
    }
  },

  stopCamera: () => {
    get().stopBroadcasting();
    if (activeMediaStream) {
      activeMediaStream.getTracks().forEach((t) => t.stop());
      activeMediaStream = null;
    }
    if (attachedVideoEl) {
      attachedVideoEl.srcObject = null;
    }
    set({ isCameraActive: false, isBroadcasting: false, fps: 0 });
  },

  toggleFacing: async () => {
    const next = get().facingMode === 'environment' ? 'user' : 'environment';
    const wasBroadcasting = get().isBroadcasting;
    const ok = await get().startCamera(next);
    if (ok && wasBroadcasting) {
      // Re-link broadcast to existing socket
    }
  },

  startBroadcasting: (serverBase: string) => {
    if (get().isBroadcasting) return;

    if (!activeMediaStream || !attachedVideoEl) {
      get().startCamera().then((ok) => {
        if (ok) get().startBroadcasting(serverBase);
      });
      return;
    }

    if (!offscreenCanvas && typeof document !== 'undefined') {
      offscreenCanvas = document.createElement('canvas');
      offscreenCanvas.width = 640;
      offscreenCanvas.height = 360;
    }

    const wsUrl = `${serverBase.replace(/^http/i, 'ws').replace(/\/+$/, '')}/ws/camera`;
    let ws: WebSocket;
    try {
      ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';
    } catch (err: any) {
      set({ error: `Cannot connect to stream server: ${err?.message}` });
      return;
    }

    broadcastWs = ws;
    lastFrameCount = 0;
    framesInSecond = 0;
    lastFpsTs = Date.now();

    ws.onopen = () => {
      set({ isBroadcasting: true, error: null });

      const ctx = offscreenCanvas?.getContext('2d');
      // Capture and push at ~18 FPS (every 55ms)
      broadcastInterval = setInterval(() => {
        if (
          !attachedVideoEl ||
          !offscreenCanvas ||
          !ctx ||
          attachedVideoEl.readyState < 2 ||
          ws.readyState !== WebSocket.OPEN
        ) {
          return;
        }

        ctx.drawImage(attachedVideoEl, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
        offscreenCanvas.toBlob(
          (blob) => {
            if (blob && ws.readyState === WebSocket.OPEN) {
              blob.arrayBuffer().then((buf) => {
                if (ws.readyState === WebSocket.OPEN) {
                  ws.send(buf);
                  framesInSecond++;
                  const now = Date.now();
                  if (now - lastFpsTs >= 1000) {
                    const currentFps = Math.round((framesInSecond * 1000) / (now - lastFpsTs) * 10) / 10;
                    set((s) => ({
                      fps: currentFps,
                      framesSent: s.framesSent + framesInSecond,
                    }));
                    framesInSecond = 0;
                    lastFpsTs = now;
                  }
                }
              });
            }
          },
          'image/jpeg',
          0.65,
        );
      }, 55);
    };

    ws.onerror = () => {
      set({ error: 'Stream WebSocket error. Ensure server is reachable.' });
    };

    ws.onclose = () => {
      if (broadcastInterval) {
        clearInterval(broadcastInterval);
        broadcastInterval = null;
      }
      set({ isBroadcasting: false, fps: 0 });
    };
  },

  stopBroadcasting: () => {
    if (broadcastInterval) {
      clearInterval(broadcastInterval);
      broadcastInterval = null;
    }
    if (broadcastWs) {
      try {
        broadcastWs.close();
      } catch {}
      broadcastWs = null;
    }
    set({ isBroadcasting: false, fps: 0 });
  },
}));
