"""Compatibility import for the shared official SuperMimic logger."""

from pathlib import Path
import sys

SHARED = Path(__file__).resolve().parents[2] / "telemetry"
if not (SHARED / "official_sm_logger.py").is_file():
    raise RuntimeError("Use the repository checkout with its shared telemetry directory")
sys.path.insert(0, str(SHARED))

from official_sm_logger import StandardLogger, create_workspace, metric_definitions

__all__ = ["StandardLogger", "create_workspace", "metric_definitions"]
