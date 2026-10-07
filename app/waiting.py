import asyncio
import json
import logging
import math
import os
import time
from contextlib import suppress
from pathlib import Path

from . import config
from .conversations import _now as _now_iso

log = logging.getLogger(__name__)

ROLLING_FILE = Path(config.ROLLING_FILE)
METRICS_FILE = Path(config.METRICS_FILE)

_rolling_lock = asyncio.Lock()
_metrics_lock = asyncio.Lock()


# ============== الإحصائيات المتراكمة ==============
def _zero_rolling_template() -> dict:
    return {"voice": {"count": 0, "avg_elapsed": 0.0}, "text": {"count": 0, "avg_elapsed": 0.0}}


async def _load_rolling() -> dict:
    def _read() -> dict:
        if not ROLLING_FILE.exists():
            return _zero_rolling_template()
        try:
            data = json.loads(ROLLING_FILE.read_text(encoding="utf-8"))
            for rtype, zero in _zero_rolling_template().items():
                data[rtype] = {**zero, **data[rtype]} if isinstance(data.get(rtype), dict) else zero
            return data
        except Exception:
            return _zero_rolling_template()

    return await asyncio.to_thread(_read)


async def _save_rolling(data: dict) -> None:
    def _write() -> None:
        ROLLING_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(ROLLING_FILE) + ".tmp"
        Path(tmp).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, ROLLING_FILE)

    await asyncio.to_thread(_write)


async def _update_rolling(request_type: str, elapsed_s: float) -> None:
    async with _rolling_lock:
        data = await _load_rolling()
        stats = data[request_type]
        n = stats["count"]
        stats["avg_elapsed"] = round((stats["avg_elapsed"] * n + elapsed_s) / (n + 1), 4)
        stats["count"] = n + 1
        await _save_rolling(data)


# ============== تسجيل الطلبات والتوقع ==============
async def log_request(*, request_type: str, elapsed_s: float, success: bool) -> None:
    rec = {
        "timestamp": _now_iso(),
        "request_type": request_type,
        "elapsed_s": round(float(elapsed_s or 0), 3),
        "success": bool(success),
    }

    def _append() -> None:
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(METRICS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    async with _metrics_lock:
        await asyncio.to_thread(_append)

    if success and elapsed_s > 0:
        try:
            await _update_rolling(request_type=request_type, elapsed_s=elapsed_s)
        except Exception:
            log.exception("failed to update rolling_stats")


async def predict_elapsed(*, request_type: str, default_s: int = 0) -> int:
    try:
        stats = (await _load_rolling()).get(request_type, {})
        if stats.get("count", 0) < config.MIN_COUNT:
            return max(int(default_s), config.COLD_START_ETA_SECONDS)
        return int(math.ceil(stats["avg_elapsed"] * (1.0 + config.SAFETY_MARGIN)))
    except Exception:
        return max(int(default_s), config.COLD_START_ETA_SECONDS)


class RequestTimer:
    def __init__(self) -> None:
        self.elapsed_s: float = 0.0
        self._start: float = 0.0

    async def __aenter__(self) -> "RequestTimer":
        self._start = time.monotonic()
        return self

    async def __aexit__(self, *_: object) -> None:
        self.elapsed_s = round(time.monotonic() - self._start, 3)


# ============== رسالة الانتظار ==============
def waiting_eta_caption(estimated_s: int) -> str:
    return f"⏳ الوقت المتوقع للرد خلال.. {estimated_s} ثانية" if estimated_s > 0 else "⏳ جاري تجهيز الرد…"


async def animate_waiting_message(message, stop_event: asyncio.Event, start_from: int = 0) -> None:
    if start_from < 2:
        while not stop_event.is_set():
            await asyncio.sleep(0.6)
        return

    for remaining in range(start_from - 1, 0, -1):
        if stop_event.is_set():
            break
        with suppress(Exception):
            await message.edit_text(f"⏳ الوقت المتوقع للرد خلال.. {remaining} ثانية")
        await asyncio.sleep(1)

    if not stop_event.is_set():
        with suppress(Exception):
            await message.edit_text("⏳ جاري ارسال الرد")


# ============== تهيئة الملفات ==============
async def ensure_runtime_sidecar_files() -> None:
    def _ensure() -> None:
        ROLLING_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not ROLLING_FILE.exists():
            ROLLING_FILE.write_text(json.dumps(_zero_rolling_template(), ensure_ascii=False, indent=2), encoding="utf-8")
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not METRICS_FILE.exists():
            METRICS_FILE.touch()

    async with _rolling_lock:
        await asyncio.to_thread(_ensure)
