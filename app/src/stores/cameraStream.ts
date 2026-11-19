/**
 * Persistent camera streaming store.
 *
 * Keeps the phone camera active and broadcasts high-definition JPEG frames
 * to the HawkEye server (/ws/camera) over WebSocket in 16:9 landscape format.
 * Defaults to Full HD (1080p) landscape with smart cropping when held in portrait
 * and native wide capture when held horizontally.
 */
import { Platform } from 'react-native';
import { create } from 'zustand';

export type FacingMode = 'environment' | 'user';
export type StreamResolution = '1080p' | '720p' | '540p';

export const RESOLUTION_CONFIG: Record<
  StreamResolution,
  { width: number; height: number; quality: number; label: string }
> = {
  '1080p': { width: 1920, height: 1080, quality: 0.75, label: 'Full HD (1080p)' },
  '720p': { width: 1280, height: 720, quality: 0.70, label: 'HD (720p)' },
  '540p': { width: 960, height: 540, quality: 0.65, label: 'SD (540p)' },
};

type CameraStreamState = {
  isCameraActive: boolean;
  isBroadcasting: boolean;
  facingMode: FacingMode;
  resolution: StreamResolution;
  isLandscape: boolean;
  fps: number;
  framesSent: number;
  error: string | null;

  // Actions
  setResolution: (res: StreamResolution) => Promise<void>;
  requestLandscape: () => Promise<void>;
  startCamera: (facing?: FacingMode, res?: StreamResolution) => Promise<boolean>;
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
let lastFpsTs = Date.now();
let framesInSecond = 0;

export const useCameraStream = create<CameraStreamState>((set, get) => ({
  isCameraActive: false,
  isBroadcasting: false,
  facingMode: 'environment',
  resolution: '1080p',
  isLandscape: true,
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

  setResolution: async (res: StreamResolution) => {
    set({ resolution: res });
    if (get().isCameraActive) {
      await get().startCamera(get().facingMode, res);
    }
  },

  requestLandscape: async () => {
    if (Platform.OS === 'web' && typeof document !== 'undefined') {
      try {
        if (!document.fullscreenElement && document.documentElement.requestFullscreen) {
          await document.documentElement.requestFullscreen().catch(() => {});
        }
        if (typeof screen !== 'undefined' && screen.orientation && (screen.orientation as any).lock) {
          await (screen.orientation as any).lock('landscape').catch(() => {});
        }
      } catch {}
    }
  },

  startCamera: async (facing, res) => {
    if (Platform.OS !== 'web') {
      set({ error: 'Device camera streaming is currently supported in web browsers.' });
      return false;
    }

    const desiredFacing = facing ?? get().facingMode;
    const desiredRes = res ?? get().resolution;
    const resCfg = RESOLUTION_CONFIG[desiredRes] || RESOLUTION_CONFIG['1080p'];

    if (activeMediaStream) {
      activeMediaStream.getTracks().forEach((t) => t.stop());
      activeMediaStream = null;
    }

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        const isSecure = typeof window !== 'undefined' ? window.isSecureContext : false;
        if (!isSecure) {
          throw new Error(
            'Browser blocked camera over HTTP. Run server with --ssl or enable chrome://flags/#unsafely-treat-insecure-origin-as-secure on phone.',
          );
        }
        throw new Error('Camera access not supported by this browser.');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: desiredFacing },
          width: { ideal: resCfg.width },
          height: { ideal: resCfg.height },
          aspectRatio: { ideal: 16 / 9 },
        },
        audio: false,
      });

      activeMediaStream = stream;
      if (attachedVideoEl) {
        attachedVideoEl.srcObject = stream;
        await attachedVideoEl.play().catch(() => {});
      }

      // Try orientation lock to landscape where supported
      if (typeof screen !== 'undefined' && screen.orientation && (screen.orientation as any).lock) {
        try {
          (screen.orientation as any).lock('landscape').catch(() => {});
        } catch {}
      }

      const vw = attachedVideoEl?.videoWidth || resCfg.width;
      const vh = attachedVideoEl?.videoHeight || resCfg.height;

      set({
        isCameraActive: true,
        facingMode: desiredFacing,
        resolution: desiredRes,
        isLandscape: vw >= vh,
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

    const currentRes = get().resolution;
    const resCfg = RESOLUTION_CONFIG[currentRes] || RESOLUTION_CONFIG['1080p'];

    if (!offscreenCanvas && typeof document !== 'undefined') {
      offscreenCanvas = document.createElement('canvas');
      offscreenCanvas.width = resCfg.width;
      offscreenCanvas.height = resCfg.height;
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

        const activeRes = get().resolution;
        const cfg = RESOLUTION_CONFIG[activeRes] || RESOLUTION_CONFIG['1080p'];
        const vw = attachedVideoEl.videoWidth || cfg.width;
        const vh = attachedVideoEl.videoHeight || cfg.height;
        const nativeLandscape = vw >= vh;

        if (get().isLandscape !== nativeLandscape) {
          set({ isLandscape: nativeLandscape });
        }

        if (offscreenCanvas.width !== cfg.width || offscreenCanvas.height !== cfg.height) {
          offscreenCanvas.width = cfg.width;
          offscreenCanvas.height = cfg.height;
        }

        if (nativeLandscape) {
          // Native landscape: phone held horizontally
          ctx.drawImage(attachedVideoEl, 0, 0, cfg.width, cfg.height);
        } else {
          // Portrait mode: smart 16:9 landscape center-crop so the stream to HawkEye
          // is always a pristine 16:9 widescreen frame without anamorphic squashing
          const cropW = vw;
          const cropH = Math.round((vw * 9) / 16);
          const cropX = 0;
          const cropY = Math.max(0, Math.round((vh - cropH) / 2));
          ctx.drawImage(
            attachedVideoEl,
            cropX, cropY, cropW, cropH,
            0, 0, cfg.width, cfg.height,
          );
        }

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
          cfg.quality,
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
