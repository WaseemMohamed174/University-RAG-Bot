import asyncio

import pytest
from pypdf import PdfWriter

from app import config, documents, gemini

REAL_429 = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota. "
    "* Quota exceeded for metric: generate_content_free_tier_requests, limit: 5, model: gemini-3.5-flash\\n"
    "Please retry in 4.658511848s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': "
    "'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '4s'}]}}"
)
REAL_503 = "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand.'}}"


class FakeErr(Exception):
    def __init__(self, msg, code):
        super().__init__(msg)
        self.code = code


@pytest.fixture
def sleeps(monkeypatch):
    rec = []

    async def fake_sleep(d):
        rec.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return rec


def test_server_delay_and_daily_quota_detection():
    assert gemini._server_delay(FakeErr(REAL_429, 429)) == pytest.approx(4.658, abs=0.01)
    assert gemini._server_delay(FakeErr(REAL_503, 503)) is None
    assert gemini._is_daily_quota(FakeErr("quota metric ...PerDay... exceeded", 429))
    assert not gemini._is_daily_quota(FakeErr(REAL_429, 429))


def test_retry_waits_for_server_requested_delay(sleeps):
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise FakeErr(REAL_429, 429)
        return "ok"

    assert asyncio.run(gemini._retry(flaky)) == "ok"
    assert sleeps and sleeps[0] >= 5.6


def test_retry_does_not_wait_on_daily_quota(sleeps):
    async def always():
        raise FakeErr("429 ... PerDay ...", 429)

    with pytest.raises(FakeErr):
        asyncio.run(gemini._retry(always))
    assert sleeps == []


def test_fallback_model_used_when_primary_overloaded(monkeypatch, sleeps):
    monkeypatch.setattr(config, "FALLBACK_MODEL", "lite")
    tried = []

    async def fake_once(model, contents, cfg):
        tried.append(model)
        if model == "main":
            raise FakeErr(REAL_503, 503)
        return "from-" + model

    monkeypatch.setattr(gemini, "_generate_once", fake_once)
    assert asyncio.run(gemini._generate("x", None, "main", attempts=2)) == "from-lite"
    assert tried == ["main", "main", "lite"]


def test_fallback_used_when_model_name_not_found(monkeypatch, sleeps):
    monkeypatch.setattr(config, "FALLBACK_MODEL", "lite")

    async def fake_once(model, contents, cfg):
        if model == "typo-model":
            raise FakeErr("404 NOT_FOUND model", 404)
        return "ok-" + model

    monkeypatch.setattr(gemini, "_generate_once", fake_once)
    assert asyncio.run(gemini._generate("x", None, "typo-model")) == "ok-lite"


def test_no_fallback_when_disabled(monkeypatch, sleeps):
    monkeypatch.setattr(config, "FALLBACK_MODEL", "lite")

    async def fake_once(model, contents, cfg):
        raise FakeErr(REAL_503, 503)

    monkeypatch.setattr(gemini, "_generate_once", fake_once)
    with pytest.raises(FakeErr):
        asyncio.run(gemini._generate("x", None, "main", fallback=False, attempts=2))


def test_rate_limiter_blocks_after_limit(monkeypatch):
    monkeypatch.setattr(config, "RPM_LIMITS", {"m": 2})
    waits = []

    async def fake_sleep(d):
        waits.append(d)
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    lim = gemini._Limiter()

    async def go():
        await lim.acquire("m")
        await lim.acquire("m")
        await lim.acquire("unlimited")
        await lim.acquire("m")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(go())
    assert len(waits) == 1 and 0 < waits[0] <= 60.1


# ============== PDF متقطع + checkpoint ==============
def make_pdf(path, pages):
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        w.write(f)


def test_pdf_failure_stops_then_resumes_only_failed_groups(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PDF_PAGES_PER_CALL", 2)
    monkeypatch.setattr(config, "PARSED_DIR", str(tmp_path / "parsed"))
    monkeypatch.setattr(config, "ALLOW_LOCAL_FALLBACK", False)
    pdf = tmp_path / "reg.pdf"
    make_pdf(str(pdf), 6)

    calls, fail = [], {"on": True}

    async def fake_transcribe(data, first_page):
        calls.append(first_page)
        if first_page == 3 and fail["on"]:
            raise FakeErr("429 RESOURCE_EXHAUSTED", 429)
        return f"=== صفحة {first_page} ===\nنص {first_page}\n\n=== صفحة {first_page + 1} ===\nنص {first_page + 1}"

    monkeypatch.setattr(gemini, "transcribe_pdf", fake_transcribe)

    with pytest.raises(RuntimeError) as ei:
        asyncio.run(documents.load_pages(str(pdf)))
    assert "3-4" in str(ei.value) and "أعد تشغيل" in str(ei.value)

    fail["on"] = False
    calls.clear()
    pages = asyncio.run(documents.load_pages(str(pdf)))
    assert calls == [3]
    assert [p for p, _ in pages] == [1, 2, 3, 4, 5, 6]
    assert not any((tmp_path / "parsed" / ".partial").iterdir())


def test_pdf_local_fallback_only_when_explicitly_allowed(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PDF_PAGES_PER_CALL", 2)
    monkeypatch.setattr(config, "PARSED_DIR", str(tmp_path / "parsed"))
    monkeypatch.setattr(config, "ALLOW_LOCAL_FALLBACK", True)
    pdf = tmp_path / "reg.pdf"
    make_pdf(str(pdf), 2)

    async def boom(*a, **k):
        raise FakeErr("503", 503)

    monkeypatch.setattr(gemini, "transcribe_pdf", boom)
    pages = asyncio.run(documents.load_pages(str(pdf)))
    assert isinstance(pages, list)
