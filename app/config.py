import os

from dotenv import load_dotenv

load_dotenv()


# ============== أدوات قراءة المتغيرات ==============
def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _rpm_limits(raw: str) -> dict[str, int]:
    pairs = (part.partition("=") for part in raw.split(",") if "=" in part)
    return {name.strip(): int(n) for name, _, n in pairs if name.strip() and n.strip().isdigit()}


# ============== المفاتيح ==============
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ADMIN_ID = _int("ADMIN_ID", 0)

# ============== Qdrant ==============
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "university_rag_v2")

# ============== موديلات Gemini ==============
FLASH_MODEL = os.getenv("FLASH_MODEL", "gemini-3.1-flash-lite")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gemini-3.1-flash-lite")
STT_MODEL = os.getenv("STT_MODEL", FLASH_MODEL)
OCR_MODEL = os.getenv("OCR_MODEL", "gemini-3.1-flash-lite")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-3.1-flash-lite")
RPM_LIMITS = _rpm_limits(os.getenv("RPM_LIMITS", "gemini-3.1-flash-lite=4"))
ALLOW_LOCAL_FALLBACK = os.getenv("ALLOW_LOCAL_FALLBACK", "0") == "1"
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = _int("EMBED_DIM", 768)

# ============== المحادثة ==============
HISTORY_LIMIT = _int("HISTORY_LIMIT", 20)
PROMPT_HISTORY = _int("PROMPT_HISTORY", 6)

# ============== البحث ==============
QUERY_TOP_K = _int("QUERY_TOP_K", 15)
CONTEXT_CHUNKS = _int("CONTEXT_CHUNKS", 4)
MIN_SCORE = _float("MIN_SCORE", 0.0)
LEXICAL_BOOST = _float("LEXICAL_BOOST", 0.15)

# ============== الدفعات ==============
BATCH_CHAR_BUDGET = _int("BATCH_CHAR_BUDGET", 32000)
BATCH_MAX_CHUNKS = _int("BATCH_MAX_CHUNKS", 10)
MAX_PARALLEL_BATCHES = _int("MAX_PARALLEL_BATCHES", 4)

# ============== ملفات JSON ==============
JSON_SKIP_KEYS = {k.strip() for k in os.getenv("JSON_SKIP_KEYS", "students,attendance_sessions,queries").split(",") if k.strip()}
MAX_LOCATIONS = _int("MAX_LOCATIONS", 3)

# ============== التقطيع ==============
CHUNK_MAX_CHARS = _int("CHUNK_MAX_CHARS", 1800)
CHUNK_MIN_CHARS = _int("CHUNK_MIN_CHARS", 500)
CHUNK_OVERLAP_CHARS = _int("CHUNK_OVERLAP_CHARS", 150)
PDF_PAGES_PER_CALL = _int("PDF_PAGES_PER_CALL", 10)

# ============== المجلدات والملفات ==============
DATA_DIR = os.getenv("DATA_DIR", "data")
PARSED_DIR = os.path.join(DATA_DIR, "parsed")
CONVERSATIONS_DIR = os.getenv("CONVERSATIONS_DIR", os.path.join(DATA_DIR, "conversations"))
LOG_FILE = os.getenv("LOG_FILE", "Logs.log")
LOG_CONSOLE_LEVEL = os.getenv("LOG_CONSOLE_LEVEL", "WARNING")

# ============== رسالة الانتظار والوقت المتوقع ==============
ROLLING_FILE = os.getenv("ROLLING_FILE", os.path.join(DATA_DIR, "rolling_stats.json"))
METRICS_FILE = os.getenv("METRICS_FILE", os.path.join(DATA_DIR, "response_metrics.jsonl"))
SAFETY_MARGIN = _float("SAFETY_MARGIN", 0.15)
MIN_COUNT = _int("MIN_COUNT", 3)
COLD_START_ETA_SECONDS = _int("COLD_START_ETA_SECONDS", 8)
