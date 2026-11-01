"""
config.py — loads secrets from a local .env file (never hard-code them in source).
Real environment variables always win over .env values.
"""
import os
from pathlib import Path

_ENV = Path(__file__).parent / ".env"


def load_env():
    if not _ENV.exists():
        return
    for line in _ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
