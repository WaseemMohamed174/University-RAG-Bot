import json
import logging
from pathlib import Path

from . import config, documents, locations
from .documents import Chunk

log = logging.getLogger(__name__)

PRIVATE_FIELDS = {"telegram_id", "password", "token", "national_id", "student_id"}

LABELS = {
    "name": "الاسم", "name_en": "الاسم بالإنجليزي", "title": "العنوان", "description": "الوصف",
    "maps_link": "رابط الخريطة", "coordinates": "الإحداثيات",
    "question": "السؤال", "answer": "الإجابة",
    "subject": "المادة", "day": "اليوم", "time": "الوقت", "location": "المكان",
    "status": "الحالة", "status_note": "ملاحظة", "links": "الروابط",
    "department": "القسم", "faculty_id": "الكلية", "faculty_member_id": "عضو هيئة التدريس",
    "email": "الإيميل", "phone": "الهاتف",
}
SECTION_LABELS = {
    "university": "الجامعة", "faculties": "الأماكن والكليات", "schedules": "الجداول",
    "faq": "أسئلة شائعة", "material_links": "روابط ومواد", "faculty_members": "أعضاء هيئة التدريس",
}
LOCATION_TEXT_KEYS = ("location", "place", "venue", "hall", "room", "address", "مكان")
LOCATION_ID_KEYS = ("location_id", "place_id")


# ============== تحويل القيم لنص ==============
def _empty(v) -> bool:
    return v in (None, "", [], {})


def _scalar(v) -> str:
    return ("نعم" if v else "لا") if isinstance(v, bool) else str(v).strip()


def _flatten(value) -> str:
    if isinstance(value, dict):
        return "، ".join(
            f"{LABELS.get(k, k)}: {_flatten(v)}" for k, v in value.items() if k not in PRIVATE_FIELDS and not _empty(v)
        )
    if isinstance(value, list):
        return "، ".join(x for x in map(_flatten, value) if x)
    return "" if _empty(value) else _scalar(value)


def _render(record: dict, by_id: dict, has_loc: bool) -> str:
    lines = []
    for k, v in record.items():
        if k == "id" or k in PRIVATE_FIELDS or _empty(v):
            continue
        if k == "coordinates" and isinstance(v, dict):
            if loc := locations.to_location({"name": "x", "coordinates": v}):
                lines.append(f"{LABELS[k]}: {loc['lat']}, {loc['lng']}")
        elif k in ("faculty_id", *LOCATION_ID_KEYS) and str(v) in by_id:
            lines.append(f"{LABELS.get(k, k)}: {locations.short_title(by_id[str(v)]['name'])}")
        elif text := _flatten(v):
            lines.append(f"{LABELS.get(k, k)}: {text}")
    if has_loc:
        lines.append("لوكيشن (موقع على الخريطة): متاح")
    return "\n".join(lines)


# ============== المواقع ==============
def collect_locations(node, skip: set[str]) -> list[dict]:
    found: list[dict] = []

    def walk(n) -> None:
        if isinstance(n, dict):
            if loc := locations.to_location(n, f"loc_auto_{len(found) + 1:03d}"):
                found.append(loc)
            for k, v in n.items():
                if k not in skip and k not in PRIVATE_FIELDS:
                    walk(v)
        elif isinstance(n, list):
            for i in n:
                walk(i)

    walk(node)
    unique: dict[str, dict] = {}
    for loc in found:
        unique.setdefault(loc["id"], loc)
    return list(unique.values())


def _record_locations(record: dict, locs: list[dict], by_id: dict) -> list[dict]:
    if own := locations.to_location(record):
        return [by_id.get(own["id"], own)]
    text_key = next((k for k in LOCATION_TEXT_KEYS if isinstance(record.get(k), str) and record[k].strip()), None)
    for k in LOCATION_ID_KEYS:
        if str(record.get(k, "")) in by_id:
            return [by_id[str(record[k])]]
    if text_key:
        if str(record.get("faculty_id", "")) in by_id:
            return [by_id[str(record["faculty_id"])]]
        if found := locations.resolve_location(record[text_key], locs):
            return [found]
    return []


# ============== السجلات والقطع ==============
def _records(node, path: list[str], skip: set[str]):
    if isinstance(node, dict):
        info = {k: v for k, v in node.items() if not isinstance(v, (dict, list))}
        if any(not _empty(v) for k, v in info.items() if k != "id"):
            yield path, info
        for k, v in node.items():
            if isinstance(v, (dict, list)) and k not in skip and k not in PRIVATE_FIELDS:
                if isinstance(v, dict) and locations.to_location(v):
                    yield path + [k], v
                else:
                    yield from _records(v, path + [k], skip)
    elif isinstance(node, list):
        if scalars := [x for x in node if not isinstance(x, (dict, list)) and not _empty(x)]:
            yield path, {"values": scalars}
        for item in node:
            if isinstance(item, dict):
                yield path, item
            elif isinstance(item, list):
                yield from _records(item, path, skip)
    elif not _empty(node):
        yield path, {"value": node}


def load_chunks(path: str, source: str) -> list[Chunk]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"ملف JSON غير صالح: {e}") from e

    skip = set(config.JSON_SKIP_KEYS)
    locs = collect_locations(data, skip)
    by_id = {loc["id"]: loc for loc in locs}

    chunks: list[Chunk] = []
    for rec_path, record in _records(data, [], skip):
        rec_locs = _record_locations(record, locs, by_id)
        text = _render(record, by_id, bool(rec_locs))
        if len(text) < 5:
            continue
        key = rec_path[-1] if rec_path else "root"
        section = SECTION_LABELS.get(key, key)
        extra = {"locations": rec_locs} if rec_locs else {}
        title = text.split("\n", 1)[0]
        for n, piece in enumerate(documents._split_long(text)):
            if n and not piece.startswith(title):
                piece = f"{title}\n{piece}"
            chunks.append(Chunk(piece, source, 0, 0, section, len(chunks), extra))
    log.info("📄 %s: %d سجل JSON، %d موقع", source, len(chunks), len(locs))
    return chunks
