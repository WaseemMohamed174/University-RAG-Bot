import html
import re

_BIDI = re.compile("[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_TASHKEEL = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")
_SPACES = re.compile(r"[ \t\u00a0]+")
_AR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})

_BOLD = re.compile(r"\*\*(?=\S)([^\n]+?)(?<=\S)\*\*")
_ITALIC = re.compile(r"(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])")
_BULLET = re.compile(r"^([ \t]*)[-*•–][ \t]+", re.M)
_TAG = re.compile(r"</?(b|i)>")


# ============== نصوص عربية ==============
def strip_bidi(text: str) -> str:
    return _BIDI.sub("", text)


def normalize_ar(text: str) -> str:
    text = _TASHKEEL.sub("", strip_bidi(text)).translate(_AR_MAP)
    return _SPACES.sub(" ", text).strip().lower()


# ============== تنسيق تيليجرام ==============
def _tags_balanced(s: str) -> bool:
    stack: list[str] = []
    for m in _TAG.finditer(s):
        if not m.group().startswith("</"):
            stack.append(m.group(1))
        elif not stack or stack.pop() != m.group(1):
            return False
    return not stack


def strip_markup(text: str) -> str:
    return _ITALIC.sub(r"\1", _BOLD.sub(r"\1", _BULLET.sub(r"\1• ", text)))


def to_telegram_html(text: str) -> str:
    text = _BULLET.sub(r"\1• ", html.escape(text.strip(), quote=False))
    out = _ITALIC.sub(r"<i>\1</i>", _BOLD.sub(r"<b>\1</b>", text))
    return out if _tags_balanced(out) else html.escape(strip_markup(text.strip()), quote=False)


def split_message(text: str, limit: int = 4000) -> list[str]:
    text = text.strip() or "..."
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) + 1 > limit:
            parts.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        parts.append(current)
    return parts
