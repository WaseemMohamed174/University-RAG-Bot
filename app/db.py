import aiosqlite

from . import config

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        username TEXT,
        full_name TEXT,
        joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('user', 'bot')),
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
    "CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, id)",
)


# ============== التنفيذ ==============
async def _run(sql: str, params: tuple = ()) -> None:
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute(sql, params)
        await db.commit()


async def init_db() -> None:
    async with aiosqlite.connect(config.DB_PATH) as db:
        for statement in _SCHEMA:
            await db.execute(statement)
        await db.commit()


# ============== المستخدمين والرسائل ==============
async def add_user(user_id: int, username: str | None, full_name: str | None) -> None:
    await _run("INSERT OR IGNORE INTO users (user_id, username, full_name) VALUES (?, ?, ?)", (user_id, username, full_name))


async def add_message(user_id: int, role: str, content: str) -> None:
    await _run("INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))


async def get_history(user_id: int, limit: int | None = None) -> list[dict]:
    async with aiosqlite.connect(config.DB_PATH) as db:
        async with db.execute(
            "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit or config.HISTORY_LIMIT),
        ) as cur:
            rows = await cur.fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


async def clear_history(user_id: int) -> None:
    await _run("DELETE FROM messages WHERE user_id = ?", (user_id,))
