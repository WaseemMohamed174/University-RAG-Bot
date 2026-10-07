import asyncio

from aiogram import Bot, Dispatcher

from . import config, vector_store, waiting
from .handlers import admin, user
from .log import setup_logging


# ============== التشغيل ==============
def build_dispatcher() -> Dispatcher:
    dp = Dispatcher()
    dp.include_routers(admin.router, user.router)
    return dp


async def main() -> None:
    for name, value in (("BOT_TOKEN", config.BOT_TOKEN), ("GEMINI_API_KEY", config.GEMINI_API_KEY)):
        if not value:
            raise SystemExit(f"{name} is missing in .env")

    await vector_store.init_collection()
    await waiting.ensure_runtime_sidecar_files()

    bot = Bot(token=config.BOT_TOKEN)
    dp = build_dispatcher()
    print("🚀 Bot is running...")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
        await vector_store.close()


if __name__ == "__main__":
    setup_logging()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 Bot stopped")
