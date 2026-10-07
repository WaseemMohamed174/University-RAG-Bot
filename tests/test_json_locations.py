import asyncio
import json
import re
import zlib

import pytest

from app import config, conversations, gemini, ingest, jsondata, locations, pipeline, vector_store
from app.schemas import FinalAnswer
from app.text_utils import normalize_ar

DIM = 768

DB = {
    "university": {
        "name": "جامعة تجريبية",
        "faculties": [
            {"id": "loc_001", "name": "كلية الطب البيطري | Faculty of Veterinary Medicine",
             "maps_link": "https://maps.app.goo.gl/vet", "coordinates": {"lat": 31.10, "lng": 30.94}},
            {"id": "loc_002", "name": "كلية التربية | Faculty of Education",
             "maps_link": "https://maps.app.goo.gl/edu", "coordinates": {"lat": 31.11, "lng": 30.95}},
            {"id": "loc_003", "name": "كلية التربية النوعية - جامعة تجريبية | Faculty of Specific Education",
             "maps_link": "https://maps.app.goo.gl/spec", "coordinates": {"lat": 31.12, "lng": 30.96}},
        ],
    },
    "students": [{"id": "stu001", "name": "طالب سري", "telegram_id": 111222333}],
    "schedules": [
        {"id": "s1", "subject": "Arduino", "day": "Tuesday", "time": "11", "location": "كلية التربية النوعية، القاعة 301"},
        {"id": "s2", "subject": "Physics", "day": "Monday", "time": "9", "location": "معمل بدون مكان معروف"},
    ],
    "faq": [{"id": "q1", "faculty_id": "loc_001", "question": "كم عدد الساعات؟", "answer": "144 ساعة"}],
    "attendance_sessions": [{"id": "a1", "note": "بيانات حضور خاصة"}],
}


def fake_vec(text: str) -> list[float]:
    v = [0.0] * DIM
    for w in re.findall(r"\w+", normalize_ar(text)):
        v[zlib.crc32(w.encode()) % DIM] += 1.0
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "QDRANT_URL", ":memory:")
    monkeypatch.setattr(config, "COLLECTION_NAME", f"j_{tmp_path.name[-8:]}")
    monkeypatch.setattr(config, "EMBED_DIM", DIM)
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", str(tmp_path / "conversations"))
    monkeypatch.setattr(config, "PARSED_DIR", str(tmp_path / "parsed"))
    monkeypatch.setattr(vector_store, "_client", None)

    async def fake_embed(texts, task_type):
        return [fake_vec(t) for t in texts]

    monkeypatch.setattr(gemini, "embed", fake_embed)


@pytest.fixture
def db_file(tmp_path):
    p = tmp_path / "database.json"
    p.write_text(json.dumps(DB, ensure_ascii=False), encoding="utf-8")
    return p


# ============== تحويل JSON لقطع ==============
def test_json_chunks_skip_private_sections(db_file):
    chunks = jsondata.load_chunks(str(db_file), "database.json")
    text = "\n".join(c.text for c in chunks)
    assert "طالب سري" not in text and "111222333" not in text
    assert "بيانات حضور خاصة" not in text
    assert "كلية الطب البيطري" in text and "Arduino" in text and "144 ساعة" in text
    assert "رابط الخريطة: https://maps.app.goo.gl/vet" in text


def test_json_location_records_carry_coordinates(db_file):
    chunks = jsondata.load_chunks(str(db_file), "database.json")
    vet = next(c for c in chunks if c.text.startswith("الاسم: كلية الطب البيطري"))
    assert vet.extra["locations"] == [
        {"id": "loc_001", "name": "كلية الطب البيطري | Faculty of Veterinary Medicine",
         "lat": 31.10, "lng": 30.94, "maps_link": "https://maps.app.goo.gl/vet"}
    ]


def test_schedule_location_text_resolves_to_building(db_file):
    chunks = jsondata.load_chunks(str(db_file), "database.json")
    known = next(c for c in chunks if "Arduino" in c.text)
    unknown = next(c for c in chunks if "Physics" in c.text)
    assert [l["id"] for l in known.extra["locations"]] == ["loc_003"]
    assert "locations" not in unknown.extra
    faq = next(c for c in chunks if "144 ساعة" in c.text)
    assert "locations" not in faq.extra
    assert "الكلية: كلية الطب البيطري" in faq.text


def test_json_skip_keys_is_configurable(db_file, monkeypatch):
    monkeypatch.setattr(config, "JSON_SKIP_KEYS", set())
    text = "\n".join(c.text for c in jsondata.load_chunks(str(db_file), "database.json"))
    assert "طالب سري" in text and "111222333" not in text


def test_json_works_on_plain_list_and_rejects_invalid(tmp_path):
    p = tmp_path / "faq.json"
    p.write_text(json.dumps([{"question": "س؟", "answer": "جواب طويل كفاية"}], ensure_ascii=False), encoding="utf-8")
    chunks = jsondata.load_chunks(str(p), "faq.json")
    assert len(chunks) == 1 and "الإجابة: جواب طويل كفاية" in chunks[0].text

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        jsondata.load_chunks(str(bad), "bad.json")


# ============== أدوات اللوكيشن ==============
def test_resolve_location_prefers_earliest_then_longest():
    locs = [
        {"id": "u", "name": "جامعة كفر الشيخ | Kafrelsheikh University", "lat": 1, "lng": 1, "maps_link": ""},
        {"id": "e", "name": "كلية الهندسة - جامعة كفر الشيخ | Faculty of Engineering", "lat": 2, "lng": 2, "maps_link": ""},
        {"id": "a", "name": "كلية الذكاء الإصطناعي - جامعة كفر الشيخ", "lat": 3, "lng": 3, "maps_link": ""},
    ]
    assert locations.resolve_location("كلية الهندسة - جامعة كفر الشيخ، قاعة 1", locs)["id"] == "e"
    assert locations.resolve_location("كلية الذكاء الاصطناعي، القاعة 301", locs)["id"] == "a"
    assert locations.resolve_location("مكان مجهول", locs) is None


def test_short_title():
    assert locations.short_title("كلية الطب | Faculty of Medicine") == "كلية الطب"


# ============== الرفع ==============
def test_ingest_json_stores_locations_and_is_idempotent(db_file, tmp_path):
    async def go():
        await vector_store.init_collection()
        n1 = await ingest.ingest_file(str(db_file), "database.json", force=True)
        n2 = await ingest.ingest_file(str(db_file), "database.json")
        assert n1 == n2 > 0
        assert (await vector_store.list_sources())["database.json"] == n1
        points, _ = await vector_store.client().scroll(config.COLLECTION_NAME, limit=100, with_payload=True)
        with_loc = [p for p in points if p.payload.get("locations")]
        assert {p.payload["locations"][0]["id"] for p in with_loc} == {"loc_001", "loc_002", "loc_003"}
        assert (tmp_path / "parsed" / "database.json.md").exists()
        await vector_store.delete_source("database.json")
        assert not await vector_store.list_sources()

    asyncio.run(go())


# ============== الخط كامل ==============
def install_llm(monkeypatch, location_ids, seen):
    async def fake_json(prompt, schema, model=None):
        assert schema is FinalAnswer
        seen["prompt"] = prompt
        return {"answer": "الكلية في المدينة الجامعية 📍\n📚 المصدر: database.json", "location_ids": location_ids}

    monkeypatch.setattr(gemini, "generate_json", fake_json)


def ask(db_file, uid=5):
    async def go():
        await vector_store.init_collection()
        await ingest.ingest_file(str(db_file), "database.json", force=True)
        reply = await pipeline.answer_message_full(uid, "فين كلية الطب البيطري؟")
        return reply, await conversations.get_history(uid)

    return asyncio.run(go())


def test_pipeline_sends_location(monkeypatch, db_file, tmp_path):
    seen = {}
    install_llm(monkeypatch, ["loc_001", "loc_999"], seen)
    reply, history = ask(db_file)
    assert [l["id"] for l in reply.locations] == ["loc_001"]
    assert reply.locations[0]["lat"] == 31.10 and reply.locations[0]["maps_link"].endswith("/vet")
    assert "موقع: loc_001" in seen["prompt"]
    saved = json.loads((tmp_path / "conversations" / "5.json").read_text(encoding="utf-8"))
    assert saved["conversations"][-1]["messages"][-1]["meta"] == {"locations": ["loc_001"]}


def test_no_location_when_model_returns_none(monkeypatch, db_file):
    install_llm(monkeypatch, [], {})
    reply, _ = ask(db_file)
    assert reply.locations == []


def test_location_not_in_context_is_rejected(monkeypatch, db_file):
    install_llm(monkeypatch, ["loc_002"], {})
    monkeypatch.setattr(config, "CONTEXT_CHUNKS", 1)
    reply, _ = ask(db_file)
    assert reply.locations == []


def test_answer_message_still_returns_plain_text(monkeypatch, db_file):
    install_llm(monkeypatch, ["loc_001"], {})

    async def go():
        await vector_store.init_collection()
        await ingest.ingest_file(str(db_file), "database.json", force=True)
        return await pipeline.answer_message(9, "فين كلية الطب البيطري؟")

    assert asyncio.run(go()).startswith("الكلية في المدينة الجامعية")
