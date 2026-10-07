import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from . import config

log = logging.getLogger(__name__)
_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


# ============== الملفات ==============
def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _path(user_id: int) -> Path:
    return Path(config.CONVERSATIONS_DIR) / f"{int(user_id)}.json"


def _load(user_id: int) -> dict:
    p = _path(user_id)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data.setdefault("conversations", [])
            return data
        except (json.JSONDecodeError, OSError):
            bad = p.with_name(f"{p.stem}.corrupt-{int(time.time())}.json")
            p.rename(bad)
            log.error("ملف المحادثة %s تالف، اتنقل لـ %s", p, bad)
    return {"user_id": int(user_id), "username": None, "full_name": None, "created_at": _now(), "conversations": []}


def _save(user_id: int, data: dict) -> None:
    p = _path(user_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


async def _locked(user_id: int, work):
    async with _locks[user_id]:
        return await asyncio.to_thread(work)


# ============== بيانات المحادثة ==============
def _apply_profile(data: dict, profile: dict | None) -> None:
    data.update({k: profile[k] for k in ("username", "full_name") if profile and profile.get(k)})


def _current(data: dict) -> dict:
    if not data["conversations"]:
        data["conversations"].append({"id": 1, "started_at": _now(), "messages": []})
    return data["conversations"][-1]


# ============== العمليات العامة ==============
async def touch_user(user_id: int, profile: dict | None = None) -> None:
    def work() -> None:
        data = _load(user_id)
        _apply_profile(data, profile)
        _current(data)
        _save(user_id, data)

    await _locked(user_id, work)


async def append_message(user_id: int, role: str, content: str, *, profile: dict | None = None, **extra) -> None:
    def work() -> None:
        data = _load(user_id)
        _apply_profile(data, profile)
        _current(data)["messages"].append(
            {"role": role, "content": content, "time": _now(), **{k: v for k, v in extra.items() if v is not None}}
        )
        _save(user_id, data)

    await _locked(user_id, work)


async def get_history(user_id: int, limit: int | None = None) -> list[dict]:
    data = await _locked(user_id, lambda: _load(user_id))
    if not data["conversations"]:
        return []
    msgs = [m for m in data["conversations"][-1]["messages"] if not m.get("error")]
    return [{"role": m["role"], "content": m["content"]} for m in msgs[-(limit or config.HISTORY_LIMIT) :]]


async def new_conversation(user_id: int) -> None:
    def work() -> None:
        data = _load(user_id)
        convs = data["conversations"]
        if convs and not convs[-1]["messages"]:
            return
        convs.append({"id": len(convs) + 1, "started_at": _now(), "messages": []})
        _save(user_id, data)

    await _locked(user_id, work)
