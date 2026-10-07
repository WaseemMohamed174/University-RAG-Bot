import argparse
import asyncio
import logging
import uuid
from pathlib import Path

from qdrant_client import models

from . import config, documents, gemini, jsondata, vector_store
from .log import setup_logging

log = logging.getLogger(__name__)
SUPPORTED = {".pdf", ".docx", ".json"}
_HELP = """رفع الملفات لقاعدة المعرفة.

الاستخدام من سطر الأوامر (من جذر المشروع):
    python -m app.ingest            # كل ملفات data/pdfs و data/docs
    python -m app.ingest --force    # إعادة قراءة الملفات حتى لو اتقرت قبل كده
والأدمن كمان يقدر يبعت الملف للبوت مباشرة.
"""


# ============== رفع ملف واحد ==============
def _save_parsed(parsed: Path, text: str) -> None:
    parsed.parent.mkdir(parents=True, exist_ok=True)
    parsed.write_text(text, encoding="utf-8")


async def ingest_file(path: str, display_name: str | None = None, force: bool = False) -> int:
    name = display_name or Path(path).name
    parsed = Path(config.PARSED_DIR) / f"{name}.md"

    if Path(path).suffix.lower() == ".json":
        chunks = jsondata.load_chunks(path, name)
        _save_parsed(parsed, "\n\n".join(f"[{c.section}]\n{c.text}" for c in chunks))
    else:
        if not force and parsed.exists() and parsed.stat().st_mtime >= Path(path).stat().st_mtime:
            log.info("📄 %s: استخدام النسخة المقروءة المحفوظة (%s)", name, parsed)
            pages = documents.parse_page_markers(parsed.read_text(encoding="utf-8"))
        else:
            log.info("📄 %s: جاري القراءة...", name)
            pages = await documents.load_pages(path)
            _save_parsed(parsed, documents.pages_to_markdown(pages))
        chunks = documents.chunk_pages(pages, name)
    if not chunks:
        return 0

    vectors = await gemini.embed([c.embed_text for c in chunks], "RETRIEVAL_DOCUMENT")
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}#{c.index}")),
            vector=v,
            payload={
                "text": c.text,
                "source": name,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "section": c.section,
                "chunk_index": c.index,
                **c.extra,
            },
        )
        for c, v in zip(chunks, vectors)
    ]
    await vector_store.delete_source(name)
    await vector_store.upsert(points)
    log.info("✅ %s: %d قطعة", name, len(points))
    return len(points)


# ============== التشغيل من سطر الأوامر ==============
async def _main() -> None:
    parser = argparse.ArgumentParser(description=_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="إعادة قراءة الملفات من الصفر")
    args = parser.parse_args()

    await vector_store.init_collection()
    files = sorted(
        p for d in ("pdfs", "docs") for p in (Path(config.DATA_DIR) / d).glob("*") if p.suffix.lower() in SUPPORTED
    )
    if not files:
        print(f"مفيش ملفات PDF/DOCX/JSON في {config.DATA_DIR}/pdfs أو {config.DATA_DIR}/docs")
        return
    total, failed = 0, []
    for f in files:
        try:
            total += await ingest_file(str(f), force=args.force)
        except Exception as e:
            failed.append(f.name)
            print(f"\n❌ {f.name}: {e}\n")
    await vector_store.close()
    print(f"\n🎉 إجمالي القطع المرفوعة: {total} من {len(files) - len(failed)}/{len(files)} ملف.")
    if failed:
        print("⚠️ ملفات لم تكتمل:", "، ".join(failed), "\n   أعد تشغيل: python -m app.ingest (هيكمل من حيث وقف)")
        raise SystemExit(1)


if __name__ == "__main__":
    setup_logging()
    asyncio.run(_main())
