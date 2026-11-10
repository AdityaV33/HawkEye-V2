"""HawkEye entry point.

    python -m hawkeye.run                      # webcam 0, default config
    python -m hawkeye.run --source phone.mp4   # a video file (looped)
    python -m hawkeye.run --source http://192.168.1.5:8080/video   # phone IP-cam
    python -m hawkeye.run --backend mock       # force offline mock reasoner

Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse

from .config import PROJECT_ROOT, load_config
from .runtime import set_config, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description="HawkEye — explainable anomaly detection")
    ap.add_argument("--config", default=None, help="Path to a YAML config")
    ap.add_argument("--source", default=None, help="Camera index, file path, or IP-cam URL")
    ap.add_argument("--backend", default=None,
                    help="Stage B reasoner: auto|claude|moondream|mock")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-identity", action="store_true", help="Disable face identity")
    ap.add_argument("--ssl", action="store_true", help="Enable HTTPS (required for phone camera without chrome flags)")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    if args.source is not None:
        cfg.ingest.source = args.source
    if args.backend is not None:
        cfg.reason.backend = args.backend
    if args.host is not None:
        cfg.server.host = args.host
    if args.port is not None:
        cfg.server.port = args.port
    if args.no_identity:
        cfg.identity.enabled = False
    set_config(cfg)

    import uvicorn

    from server.app import app  # imported after config is set

    ssl_kwargs = {}
    if args.ssl:
        cert_file = PROJECT_ROOT / "data" / "certs" / "cert.pem"
        key_file = PROJECT_ROOT / "data" / "certs" / "key.pem"
        if cert_file.is_file() and key_file.is_file():
            ssl_kwargs = {"ssl_certfile": str(cert_file), "ssl_keyfile": str(key_file)}

    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="info", **ssl_kwargs)


if __name__ == "__main__":
    main()
