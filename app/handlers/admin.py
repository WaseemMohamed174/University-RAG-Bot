import logging
import tempfile
from pathlib import Path

from aiogram import F, Router, types
from aiogram.filters import Command, CommandObject

from .. import config, ingest, vector_store

log = logging.getLogger(__name__)
router = Router()
router.message.filter(F.from_user.id == config.ADMIN_ID)

MAX_FILE_MB = 20


# ============== رفع الملفات ==============
@router.message(F.document)
async def on_document(message: types.Message) -> None:
    doc = message.document
    name = doc.file_name or "file"
    suffix = Path(name).suffix.lower()
    if suffix not in ingest.SUPPORTED:
        await message.answer("⚠️ ابعت ملف PDF أو Word (.docx) أو JSON (.json) بس.")
        return
    if doc.file_size and doc.file_size > MAX_FILE_MB * 1024 * 1024:
        await message.answer(f"⚠️ الملف أكبر من {MAX_FILE_MB}MB. ضعه في data/pdfs (أو data/docs لملفات JSON) وشغّل: python -m app.ingest")
        return

    await message.answer("⏳ جاري قراءة الملف ورفعه... (ممكن ياخد كام دقيقة للملفات الكبيرة)")
    tmp_path = None
    try:
        buf = await message.bot.download(doc)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(buf.read())
            tmp_path = tmp.name
        n = await ingest.ingest_file(tmp_path, display_name=name, force=True)
        await message.answer(f"✅ تمت إضافة «{name}» ({n} قطعة)." if n else "⚠️ الملف فاضي أو مفيش نص قابل للقراءة.")
    except Exception as e:
        log.exception("ingest failed")
        await message.answer(f"❌ فشل الرفع: {str(e)[:400]}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ============== إدارة الملفات ==============
@router.message(Command("files"))
async def cmd_files(message: types.Message) -> None:
    if not (sources := await vector_store.list_sources()):
        await message.answer("القاعدة فاضية.")
        return
    await message.answer("📚 الملفات:\n" + "\n".join(f"• {s} ({n} قطعة)" for s, n in sorted(sources.items())))


@router.message(Command("delete"))
async def cmd_delete(message: types.Message, command: CommandObject) -> None:
    if not command.args:
        await message.answer("الاستخدام: /delete اسم_الملف.pdf")
        return
    name = command.args.strip()
    if name not in await vector_store.list_sources():
        await message.answer("مفيش ملف بالاسم ده. استخدم /files.")
        return
    await vector_store.delete_source(name)
    await message.answer(f"🗑️ اتحذف «{name}».")
