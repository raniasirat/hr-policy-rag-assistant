"""Start the HR Assistant (API + web interface) with one command.

Usage (from the project root):
    python scripts/run_server.py
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402

from backend.dependencies import load_api_settings  # noqa: E402
from src.config import ConfigurationError, load_settings  # noqa: E402


def main() -> int:
    try:
        load_settings()  # loads .env and validates the RAG settings
        api = load_api_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    print(f"HR Assistant: http://{api.host}:{api.port}  (Ctrl+C to stop)")
    uvicorn.run("backend.main:app", host=api.host, port=api.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
