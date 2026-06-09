import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    fmt: str = "%(asctime)s | %(name)-40s | %(levelname)-7s | %(message)s",
    datefmt: str = "%H:%M:%S",
) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )
    logging.getLogger("livemesh").setLevel(getattr(logging, level.upper()))
