import asyncio
import logging
import re
from dataclasses import dataclass, field

from . import config, conversations, gemini, prompts, vector_store
from .schemas import BatchNotes, FinalAnswer
from .text_utils import normalize_ar

log = logging.getLogger(__name__)

MSG_BUSY = "الخدمة مشغولة حاليًا 🙏 جرّب تاني بعد دقيقة."
MSG_NOT_FOUND = (
    "ملقيتش إجابة لسؤالك في الملفات المتاحة عندي 😕\n"
    "جرّب تعيد صياغة السؤال بتفاصيل أكتر، أو راجع شؤون الطلاب بالكلية."
)
NO_CONTEXT = "(مفيش نصوص ذات صلة في الملفات)"
FOLLOW_UP_MAX_CHARS = 60
_LOC_KEYS = ("id", "name", "lat", "lng", "maps_link")

_STOP = {normalize_ar(w) for w in (
    "في من عن على الى إلى هو هي ايه اية اي ازاي إزاي فين كام ليه امتى إمتى ممكن عايز عاوز عايزة لو بس يعني طيب طب "
    "مع ده دي دا هل ماذا كيف اين متى لماذا الي التي الذي هذا هذه").split()}


@dataclass
class Hit:
    id: str
    text: str
    source: str
    page_start: int
    page_end: int
    section: str
    score: float
    locations: list[dict] = field(default_factory=list)

    @property
    def pages(self) -> str:
        if not self.page_start:
            return ""
        if self.page_end and self.page_end != self.page_start:
            return f"ص {self.page_start}-{self.page_end}"
        return f"ص {self.page_start}"


@dataclass
class Reply:
    text: str
    locations: list[dict] = field(default_factory=list)


# ============== المحادثة والاستعلامات ==============
def format_conversation(history: list[dict]) -> str:
    names = {"user": "الطالب", "bot": "البوت"}
    return "\n".join(f"{names.get(m['role'], m['role'])}: {m['content']}" for m in history) or "(بداية المحادثة)"


def build_queries(history: list[dict], user_text: str) -> list[str]:
    prev = [m["content"] for m in history[:-1] if m["role"] == "user"]
    return [user_text, f"{prev[-1]} {user_text}"] if prev and len(user_text) <= FOLLOW_UP_MAX_CHARS else [user_text]


def lexical_terms(text: str) -> list[str]:
    return list({w for w in re.findall(r"\w+", normalize_ar(text)) if len(w) >= 3 and w not in _STOP})


def lexical_score(text: str, terms: list[str]) -> float:
    norm = normalize_ar(text)
    return sum(t in norm for t in terms) / len(terms) if terms else 0.0


# ============== البحث في Qdrant ==============
def _to_hit(point, adjusted: float) -> Hit:
    p = point.payload or {}
    return Hit(
        id=str(point.id),
        text=p.get("text", ""),
        source=p.get("source", "?"),
        page_start=int(p.get("page_start") or 0),
        page_end=int(p.get("page_end") or 0),
        section=p.get("section", ""),
        score=adjusted,
        locations=[
            {k: loc.get(k) for k in _LOC_KEYS}
            for loc in (p.get("locations") or [])
            if isinstance(loc, dict) and loc.get("id") and loc.get("lat") is not None and loc.get("lng") is not None
        ],
    )


def interleave(ranked_lists: list[list[Hit]], limit: int) -> list[Hit]:
    out, seen = [], set()
    for rank in range(max((len(r) for r in ranked_lists), default=0)):
        for lst in ranked_lists:
            if len(out) >= limit:
                return out
            if rank < len(lst) and lst[rank].id not in seen:
                seen.add(lst[rank].id)
                out.append(lst[rank])
    return out


async def retrieve(queries: list[str]) -> list[Hit]:
    vectors = await gemini.embed(queries, "RETRIEVAL_QUERY")
    results = await asyncio.gather(*(vector_store.search(v, config.QUERY_TOP_K) for v in vectors))
    terms = lexical_terms(queries[0])
    ranked = [
        sorted(
            (_to_hit(p, p.score + config.LEXICAL_BOOST * lexical_score((p.payload or {}).get("text", ""), terms)) for p in points),
            key=lambda h: h.score,
            reverse=True,
        )
        for points in results
    ]
    return interleave(ranked, config.CONTEXT_CHUNKS)


# ============== تجهيز السياق ==============
def _meta(*items: str) -> str:
    return " | ".join(x for x in items if x)


def format_chunks(batch: list[Hit]) -> str:
    parts = []
    for n, h in enumerate(batch, 1):
        loc_ids = "، ".join(loc["id"] for loc in h.locations)
        meta = _meta(f"ملف: {h.source}", h.pages, f"القسم: {h.section}" if h.section else "", f"موقع: {loc_ids}" if loc_ids else "")
        parts.append(f"[قطعة {n} | {meta}]\n{h.text}")
    return "\n\n---\n\n".join(parts)


def make_batches(hits: list[Hit]) -> list[list[Hit]]:
    batches: list[list[Hit]] = []
    cur: list[Hit] = []
    size = 0
    for h in hits:
        if cur and (size + len(h.text) > config.BATCH_CHAR_BUDGET or len(cur) >= config.BATCH_MAX_CHUNKS):
            batches.append(cur)
            cur, size = [], 0
        cur.append(h)
        size += len(h.text)
    if cur:
        batches.append(cur)
    return batches


async def notes_for_batch(no: int, batch: list[Hit], question: str, convo: str) -> list[dict] | None:
    try:
        raw = await gemini.generate_json(
            prompts.notes_prompt(question, convo, format_chunks(batch)), BatchNotes, config.FLASH_MODEL
        )
    except Exception:
        log.exception("notes batch %d failed", no)
        return None
    batch_loc_ids = {loc["id"] for h in batch for loc in h.locations}
    return [
        {
            "fact": fact,
            "evidence": str(n.get("evidence") or "").strip(),
            "source": str(n.get("source") or "").strip(),
            "location_ids": [i for i in (str(x).strip() for x in (n.get("location_ids") or [])) if i in batch_loc_ids],
        }
        for n in raw.get("notes") or []
        if (fact := str(n.get("fact") or "").strip())
    ]


def format_notes(notes: list[dict]) -> str:
    parts = []
    for n, x in enumerate(notes, 1):
        loc_ids = "، ".join(x["location_ids"])
        meta = _meta(f"ملف: {x['source'] or '؟'}", f"موقع: {loc_ids}" if loc_ids else "")
        parts.append(f"[معلومة {n} | {meta}]\n{x['fact']}\nالدليل: {x['evidence'] or '—'}")
    return "\n\n---\n\n".join(parts)


async def build_context(hits: list[Hit], question: str, convo: str) -> tuple[str, set[str]] | None:
    if not hits:
        return NO_CONTEXT, set()
    if sum(len(h.text) for h in hits) <= config.BATCH_CHAR_BUDGET:
        return format_chunks(hits), {loc["id"] for h in hits for loc in h.locations}

    batches = make_batches(hits)
    log.info("context too large: %d chunks -> %d batches", len(hits), len(batches))
    sem = asyncio.Semaphore(config.MAX_PARALLEL_BATCHES)

    async def worker(no: int, batch: list[Hit]):
        async with sem:
            return await notes_for_batch(no, batch, question, convo)

    results = await asyncio.gather(*(worker(i, b) for i, b in enumerate(batches, 1)))
    if all(r is None for r in results):
        return None
    if not (notes := [n for r in results if r for n in r]):
        return NO_CONTEXT, set()
    return format_notes(notes), {i for n in notes for i in n["location_ids"]}


# ============== الرد واللوكيشن ==============
def pick_locations(ids: list, allowed: set[str], loc_map: dict[str, dict]) -> list[dict]:
    out: list[dict] = []
    for i in (str(x).strip() for x in ids or []):
        if i in allowed and i in loc_map and loc_map[i] not in out:
            out.append(loc_map[i])
    return out[: config.MAX_LOCATIONS]


async def run_full(history: list[dict], user_text: str) -> Reply:
    recent = history[-config.PROMPT_HISTORY :]
    convo = format_conversation(recent[:-1])

    hits = await retrieve(build_queries(history, user_text))
    log.info("retrieved %d chunks", len(hits))
    if (ctx := await build_context(hits, user_text, convo)) is None:
        return Reply(MSG_BUSY)
    context, allowed = ctx

    try:
        raw = await gemini.generate_json(prompts.answer_prompt(convo, user_text, context), FinalAnswer, config.ANSWER_MODEL)
    except Exception:
        log.exception("answer failed")
        return Reply(MSG_BUSY)

    if not (text := str(raw.get("answer") or "").strip()):
        return Reply(MSG_NOT_FOUND)
    loc_map = {loc["id"]: loc for h in hits for loc in h.locations}
    return Reply(text, pick_locations(raw.get("location_ids"), allowed, loc_map))


async def run(history: list[dict], user_text: str) -> str:
    return (await run_full(history, user_text)).text


# ============== نقطة الدخول ==============
async def answer_message_full(user_id: int, user_text: str, *, profile: dict | None = None, **extra) -> Reply:
    await conversations.append_message(user_id, "user", user_text, profile=profile, **extra)
    reply = await run_full(await conversations.get_history(user_id), user_text)
    await conversations.append_message(
        user_id,
        "bot",
        reply.text,
        error=reply.text == MSG_BUSY or None,
        meta={"locations": [loc["id"] for loc in reply.locations]} if reply.locations else None,
    )
    return reply


async def answer_message(user_id: int, user_text: str, **kwargs) -> str:
    return (await answer_message_full(user_id, user_text, **kwargs)).text
