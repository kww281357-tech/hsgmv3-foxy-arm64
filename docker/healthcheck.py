"""Liveness only: healthy does NOT mean a robot is connected or safe to move."""
import json
import os
import time
from pathlib import Path

status = json.loads(Path(os.environ.get("HSGM_STATUS_PATH", "/tmp/hsgm-status.json")).read_text())
raise SystemExit(0 if 0 <= time.monotonic() - status["heartbeat_monotonic"] < 20 else 1)
