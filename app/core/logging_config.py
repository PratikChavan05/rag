from __future__ import annotations
import logging
import sys

def configure_logging(level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(format=fmt, level=log_level, stream=sys.stdout, force=True)
    # Suppress noisy third-party loggers
    for name in ("httpx", "httpcore", "pdfminer", "PIL", "botocore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
