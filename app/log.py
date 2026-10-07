import logging
import sys
from logging.handlers import RotatingFileHandler

from . import config

_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


# ============== إعداد اللوج ==============
def setup_logging(console_level: str | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    file_handler = RotatingFileHandler(config.LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_FMT))
    file_handler.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    console.setLevel((console_level or config.LOG_CONSOLE_LEVEL).upper())

    for handler in (file_handler, console):
        root.addHandler(handler)
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.ERROR)

    previous_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            logging.getLogger("crash").critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        previous_hook(exc_type, exc, tb)

    sys.excepthook = _hook
