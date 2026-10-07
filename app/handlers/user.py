import asyncio
import logging
from collections import defaultdict
from contextlib import suppress

from aiogram import F, Router, types
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup
from aiogram.utils.chat_action import ChatActionSender

from .. import conversations, gemini, pipeline, waiting
from ..locations import short_title
from ..text_utils import split_message, strip_markup, to_telegram_html

log = logging.getLogger(__name__)
router = Router()

BTN_NEW = "🆕 محادثة جديدة"
BTN_HELP = "ℹ️ مساعدة"
MAX_VOICE_SECONDS = 120
TG_CHUNK = 3500

HELP_TEXT = (
    "🎓 أنا مساعد الجامعة الذكي.\n\n"
    "• اسألني نصًا أو بالصوت 🎤 عن اللوائح والجداول.\n"
    "• اسألني عن أي مكان (مثلًا: فين كلية الهندسة؟) وأبعتلك اللوكيشن 📍.\n"
    "• بفتكر كلامنا في نفس المحادثة، فتقدر تسأل 'وبالنسبة للفصل التاني؟' وأنا فاهم.\n"
    "• لو عايز تبدأ موضوع جديد اضغط 🆕 محادثة جديدة (أو /new)."
)

_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


# ============== الإرسال ==============
def main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=BTN_NEW), KeyboardButton(text=BTN_HELP)]], resize_keyboard=True
    )


async def send_formatted(message: types.Message, text: str) -> None:
    for part in split_message(text, limit=TG_CHUNK):
        try:
            await message.answer(to_telegram_html(part), parse_mode=ParseMode.HTML)
        except TelegramBadRequest:
            log.warning("HTML rejected by Telegram; sending plain text", exc_info=True)
            await message.answer(strip_markup(part), parse_mode=None)


async def send_locations(message: types.Message, locs: list[dict]) -> None:
    for loc in locs:
        link = loc.get("maps_link") or ""
        markup = (
            InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🗺️ افتح في خرائط جوجل", url=link)]])
            if link.startswith("http")
            else None
        )
        try:
            await message.answer_venue(
                latitude=loc["lat"],
                longitude=loc["lng"],
                title=short_title(loc["name"]),
                address=link or f"{loc['lat']:.6f}, {loc['lng']:.6f}",
                reply_markup=markup,
            )
        except Exception:
            log.exception("send venue failed for %s", loc.get("id"))
            try:
                await message.answer_location(latitude=loc["lat"], longitude=loc["lng"], reply_markup=markup)
            except Exception:
                log.exception("send location failed for %s", loc.get("id"))


# ============== رسالة الانتظار ==============
async def _start_waiting(message: types.Message, request_type: str) -> tuple:
    estimated_s = await waiting.predict_elapsed(request_type=request_type)
    waiting_msg = await message.answer(waiting.waiting_eta_caption(estimated_s), reply_to_message_id=message.message_id)
    stop_event = asyncio.Event()
    task = asyncio.create_task(waiting.animate_waiting_message(waiting_msg, stop_event, estimated_s))
    await asyncio.sleep(0)
    return waiting_msg, stop_event, task


async def _stop_waiting(waiting_msg: types.Message, stop_event: asyncio.Event, task: asyncio.Task) -> None:
    stop_event.set()
    with suppress(Exception):
        await task
    with suppress(TelegramBadRequest):
        await waiting_msg.delete()


# ============== معالجة الأسئلة ==============
async def process_question(message: types.Message, text: str) -> None:
    async with _locks[message.from_user.id]:
        waiting_state = await _start_waiting(message, "text")
        success = True
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            try:
                async with waiting.RequestTimer() as rt:
                    reply = await pipeline.answer_message_full(message.from_user.id, text)
            except Exception:
                log.exception("pipeline error")
                reply, success = pipeline.Reply(pipeline.MSG_BUSY), False

        await waiting.log_request(request_type="text", elapsed_s=rt.elapsed_s, success=success)
        await _stop_waiting(*waiting_state)
        await send_formatted(message, reply.text)
        await send_locations(message, reply.locations)


# ============== الأوامر ==============
@router.message(CommandStart())
async def cmd_start(message: types.Message) -> None:
    u = message.from_user
    await conversations.touch_user(u.id, {"username": u.username, "full_name": u.full_name})
    await message.answer(
        "أهلًا بيك في المساعد الجامعي الذكي! 🎓\nاسأل أي سؤال عن اللوائح أو الجداول، نصًا أو بالصوت 🎤",
        reply_markup=main_keyboard(),
    )


@router.message(Command("help"))
@router.message(F.text == BTN_HELP)
async def cmd_help(message: types.Message) -> None:
    await message.answer(HELP_TEXT)


@router.message(Command("new"))
@router.message(F.text == BTN_NEW)
async def cmd_new(message: types.Message) -> None:
    await conversations.new_conversation(message.from_user.id)
    await message.answer("تمام ✅ بدأنا محادثة جديدة. اسألني اللي عايزه.")


@router.message(Command("id"))
async def cmd_id(message: types.Message) -> None:
    await message.answer(f"ID بتاعك: {message.from_user.id}")


# ============== الصوت والنص ==============
@router.message(F.voice)
async def on_voice(message: types.Message) -> None:
    voice = message.voice
    if voice.duration and voice.duration > MAX_VOICE_SECONDS:
        await message.answer(f"التسجيل طويل ⏱️ ابعت تسجيل أقصر من {MAX_VOICE_SECONDS} ثانية.")
        return

    waiting_state = await _start_waiting(message, "voice")
    try:
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            async with waiting.RequestTimer() as rt:
                buf = await message.bot.download(voice)
                text = await gemini.transcribe_audio(buf.read(), voice.mime_type or "audio/ogg")
            elapsed = rt.elapsed_s
    except Exception:
        log.exception("voice transcription error")
        await waiting.log_request(request_type="voice", elapsed_s=0.0, success=False)
        await _stop_waiting(*waiting_state)
        await message.answer("معرفتش أسمع التسجيل 😕 جرّب تبعته تاني أو اكتب سؤالك.")
        return

    await waiting.log_request(request_type="voice", elapsed_s=elapsed, success=True)
    await _stop_waiting(*waiting_state)

    if not text or "[غير مفهوم]" in text:
        await message.answer("مقدرتش أفهم الكلام في التسجيل 😕 جرّب تتكلم بوضوح أكتر أو اكتب سؤالك.")
        return

    await message.answer(f"🎤 سمعتك بتقول:\n{text}")
    await process_question(message, text)


@router.message(F.text & ~F.text.startswith("/"))
async def on_text(message: types.Message) -> None:
    await process_question(message, message.text)
