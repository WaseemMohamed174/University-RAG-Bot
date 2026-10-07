import asyncio
import json
import logging
import math
import re
import time
from collections import deque

from google import genai
from google.genai import types

from . import config

log = logging.getLogger(__name__)

_client: genai.Client | None = None
_TRANSIENT_CODES = {429, 500, 502, 503, 504}
_TRANSIENT_WORDS = ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED")
_DELAY_PATTERNS = (re.compile(r"retry in ([\d.]+)\s*s", re.I), re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)s"))
EMBED_BATCH = 100


# ============== العميل ==============
def client() -> genai.Client:
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is missing in .env")
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


# ============== الأخطاء المؤقتة وإعادة المحاولة ==============
def _is_transient(e: Exception) -> bool:
    return getattr(e, "code", None) in _TRANSIENT_CODES or any(w in str(e) for w in _TRANSIENT_WORDS)


def _is_daily_quota(e: Exception) -> bool:
    text = str(e)
    return "PerDay" in text or "per day" in text.lower()


def _is_missing_model(e: Exception) -> bool:
    return getattr(e, "code", None) == 404 or "NOT_FOUND" in str(e)


def _server_delay(e: Exception) -> float | None:
    return next((float(m.group(1)) for pat in _DELAY_PATTERNS if (m := pat.search(str(e)))), None)


async def _retry(make_call, attempts: int = 4, base_delay: float = 2.0):
    for i in range(attempts):
        try:
            return await make_call()
        except Exception as e:
            if not _is_transient(e) or _is_daily_quota(e) or i == attempts - 1:
                raise
            delay = min(max(base_delay * (2**i), (_server_delay(e) or 0) + 1), 90)
            log.warning("Gemini %s، إعادة المحاولة بعد %.0f ثانية (%d/%d)", getattr(e, "code", "?"), delay, i + 1, attempts - 1)
            await asyncio.sleep(delay)


# ============== محدّد المعدّل ==============
class _Limiter:
    def __init__(self) -> None:
        self._stamps: dict[str, deque] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def acquire(self, model: str) -> None:
        rpm = config.RPM_LIMITS.get(model)
        if not rpm:
            return
        async with self._locks.setdefault(model, asyncio.Lock()):
            stamps = self._stamps.setdefault(model, deque())
            while True:
                now = time.monotonic()
                while stamps and now - stamps[0] >= 60:
                    stamps.popleft()
                if len(stamps) < rpm:
                    stamps.append(now)
                    return
                await asyncio.sleep(stamps[0] + 60 - now + 0.05)


_limiter = _Limiter()


# ============== الاستدعاء الأساسي ==============
def _cfg(**kw) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True), **kw)


async def _generate_once(model: str, contents, cfg: types.GenerateContentConfig):
    await _limiter.acquire(model)
    return await client().aio.models.generate_content(model=model, contents=contents, config=cfg)


async def _generate(contents, cfg, model: str, *, fallback: bool = True, attempts: int = 4):
    use_fallback = fallback and config.FALLBACK_MODEL and config.FALLBACK_MODEL != model
    candidates = [model, config.FALLBACK_MODEL] if use_fallback else [model]
    last: Exception | None = None
    for m in candidates:
        try:
            return await _retry(lambda m=m: _generate_once(m, contents, cfg), attempts)
        except Exception as e:
            if not (_is_transient(e) or _is_missing_model(e)):
                raise
            if _is_missing_model(e):
                log.error("الموديل '%s' مش موجود. راجع اسمه في .env", m)
            last = e
            if m != candidates[-1]:
                log.warning("الموديل %s مش متاح (%s) -> بجرّب %s", m, getattr(e, "code", "?"), candidates[-1])
    raise last


# ============== الدوال العامة ==============
def _l2(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


async def embed(texts: list[str], task_type: str) -> list[list[float]]:
    cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=config.EMBED_DIM)

    async def call(batch: list[str]):
        await _limiter.acquire(config.EMBED_MODEL)
        return await client().aio.models.embed_content(model=config.EMBED_MODEL, contents=batch, config=cfg)

    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = await _retry(lambda b=texts[i : i + EMBED_BATCH]: call(b), attempts=5)
        out.extend(_l2(list(e.values)) for e in resp.embeddings)
    return out


def parse_json(text: str | None) -> dict:
    if not text:
        raise ValueError("empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if not (m := re.search(r"\{.*\}", text, re.DOTALL)):
            raise ValueError(f"no JSON in response: {text[:200]!r}")
        return json.loads(m.group())


async def generate_json(prompt: str, schema, model: str | None = None) -> dict:
    cfg = _cfg(response_mime_type="application/json", response_schema=schema)
    return parse_json((await _generate(prompt, cfg, model or config.FLASH_MODEL)).text)


async def generate_text(prompt: str, model: str | None = None) -> str:
    return ((await _generate(prompt, _cfg(), model or config.FLASH_MODEL)).text or "").strip()


# ============== الصوت والـ PDF ==============
async def transcribe_audio(data: bytes, mime_type: str = "audio/ogg") -> str:
    prompt = (
        "اكتب النص المنطوق في هذا التسجيل حرفيًا كما قيل (عربي مصري غالبًا)، "
        "بدون أي إضافات أو شرح. لو مفيش كلام مفهوم اكتب: [غير مفهوم]"
    )
    contents = [types.Part.from_bytes(data=data, mime_type=mime_type), prompt]
    return ((await _generate(contents, _cfg(), config.STT_MODEL)).text or "").strip()


async def transcribe_pdf(pdf_bytes: bytes, first_page: int) -> str:
    prompt = f"""انسخ محتوى صفحات الـ PDF المرفقة حرفيًا وبدقة كاملة.
القواعد:
- لا تلخّص ولا تترجم ولا تصحّح ولا تضف شيئًا. انقل النص العربي كما هو (انتبه لحروف لا/لأ/لإ).
- حافظ على العناوين بصيغة Markdown (#)، وعلى القوائم، وعلى أي جداول بصيغة Markdown tables مع الحفاظ على كل الخلايا.
- اترك رقم الصفحة المطبوع في أسفل/أعلى الصفحة ولا تكتبه.
- ابدأ كل صفحة بسطر منفصل بالشكل: === صفحة N === حيث N هو رقم الصفحة الفعلي في الملف الأصلي. أول صفحة في هذا الجزء رقمها {first_page}، والتالية {first_page + 1} وهكذا.
- أرجع النص فقط بدون أي مقدمة أو خاتمة."""
    contents = [types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt]
    return (await _generate(contents, _cfg(), config.OCR_MODEL, fallback=False, attempts=6)).text or ""
