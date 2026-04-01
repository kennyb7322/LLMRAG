"""Centralized logging for LLMRAG pipeline."""

import logging
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Structured JSON log output."""
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
            "extra": getattr(record, "extra_data", {}),
        })


def setup_logger(
    name: str = "llmrag",
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = True,
) -> logging.Logger:
    """Create and configure a pipeline logger."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    if json_format:
        console.setFormatter(JSONFormatter())
    else:
        console.setFormatter(logging.Formatter(
            "%(asctime)s │ %(levelname)-8s │ %(module)s │ %(message)s"
        ))
    logger.addHandler(console)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(JSONFormatter())
        logger.addHandler(fh)

    return logger


# Default logger instance
log = setup_logger(json_format=False)
