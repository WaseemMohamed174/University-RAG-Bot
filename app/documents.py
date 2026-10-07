import asyncio
import io
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader, PdfWriter

from . import config, gemini
from .text_utils import strip_bidi

log = logging.getLogger(__name__)

PAGE_MARK = re.compile(r"^=== صفحة (\d+) ===[ \t]*$", re.M)
_HEADING = re.compile(r"^\s*(#+\s|(ال)?ماده\s*\(?\s*[\d٠-٩]+|(ال)?مادة\s*\(?\s*[\d٠-٩]+|الباب\s|الفصل\s)")
_DOT_LEADERS = re.compile(r"[.…·]{4,}")


@dataclass
class Chunk:
    text: str
    source: str
    page_start: int
    page_end: int
    section: str
    index: int
    extra: dict = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        head = f"{self.source} — {self.section}" if self.section else self.source
        return f"{head}\n{self.text}"


# ============== علامات الصفحات ==============
def parse_page_markers(md: str, default_page: int = 0) -> list[tuple[int, str]]:
    parts = PAGE_MARK.split(md)
    pages = [(default_page, parts[0])] + [(int(p), t) for p, t in zip(parts[1::2], parts[2::2])]
    return [(p, t.strip()) for p, t in pages if t.strip()]


def pages_to_markdown(pages: list[tuple[int, str]]) -> str:
    return "\n\n".join(f"=== صفحة {p} ===\n{t}" for p, t in pages)


# ============== قراءة PDF ==============
async def _load_pdf(path: str) -> list[tuple[int, str]]:
    reader = PdfReader(path)
    total, step, st = len(reader.pages), config.PDF_PAGES_PER_CALL, Path(path).stat()
    ck_dir = Path(config.PARSED_DIR) / ".partial" / f"{Path(path).name}-{int(st.st_mtime)}-{st.st_size}-{step}"
    ck_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(2)

    async def do_group(start: int, end: int) -> list[tuple[int, str]]:
        ck = ck_dir / f"{start:04d}.md"
        if ck.exists() and (cached := parse_page_markers(ck.read_text(encoding="utf-8"), start + 1)):
            return cached
        writer, buf = PdfWriter(), io.BytesIO()
        for page in reader.pages[start:end]:
            writer.add_page(page)
        writer.write(buf)
        async with sem:
            try:
                pages = parse_page_markers(await gemini.transcribe_pdf(buf.getvalue(), start + 1), start + 1)
                if not pages:
                    raise ValueError("Gemini رجّع نص فاضي")
            except Exception as e:
                if not config.ALLOW_LOCAL_FALLBACK:
                    raise
                log.warning("قراءة الصفحات %d-%d فشلت (%s)؛ استخدام الاستخراج المحلي (جودة العربي أقل)", start + 1, end, e)
                return [(i + 1, strip_bidi(reader.pages[i].extract_text() or "").strip()) for i in range(start, end)]
        ck.write_text(pages_to_markdown(pages), encoding="utf-8")
        log.info("   ✓ صفحات %d-%d", start + 1, end)
        return pages

    groups = [(s, min(s + step, total)) for s in range(0, total, step)]
    results = await asyncio.gather(*(do_group(s, e) for s, e in groups), return_exceptions=True)
    failed = [(g, r) for g, r in zip(groups, results) if isinstance(r, Exception)]
    if failed:
        ranges = "، ".join(f"{s + 1}-{e}" for (s, e), _ in failed)
        reason = str(failed[0][1]).replace("\n", " ")[:200]
        raise RuntimeError(
            f"فشلت قراءة الصفحات {ranges} ({reason}). "
            f"الأجزاء اللي نجحت اتحفظت، أعد تشغيل نفس الأمر وهيكمل من حيث وقف."
        )
    shutil.rmtree(ck_dir, ignore_errors=True)
    return [(p, t) for group in results for p, t in group if t]


# ============== قراءة DOCX ==============
def _table_md(table: Table) -> str:
    rows = [
        list({id(c._tc): c.text.strip().replace("\n", " ").replace("|", "/") for c in row.cells}.values())
        for row in table.rows
    ]
    if not rows:
        return ""
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([lines[0], "| " + " | ".join("---" for _ in rows[0]) + " |", *lines[1:]])


def _load_docx(path: str) -> list[tuple[int, str]]:
    doc = Document(path)
    blocks: list[str] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, doc)
            if not (text := para.text.strip()):
                continue
            try:
                style = (para.style.name or "").lower()
            except Exception:
                style = ""
            blocks.append(f"# {text}" if style.startswith(("heading", "عنوان")) or style == "title" else text)
        elif tag == "tbl" and (md := _table_md(Table(child, doc))):
            blocks.append(md)
    text = "\n\n".join(blocks)
    return [(0, text)] if text.strip() else []


async def load_pages(path: str) -> list[tuple[int, str]]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return await _load_pdf(path)
    if ext == ".docx":
        return _load_docx(path)
    raise ValueError(f"نوع ملف غير مدعوم: {ext}")


# ============== التقطيع ==============
def _is_heading(block: str) -> bool:
    return bool(_HEADING.match(block.split("\n", 1)[0]))


def _heading_label(block: str) -> str:
    return block.split("\n", 1)[0].lstrip("# ").strip()[:100]


def _split_table(block: str) -> list[str]:
    lines = block.split("\n")
    header, body = lines[:2], lines[2:]
    pieces, cur = [], list(header)
    for line in body:
        if len("\n".join(cur)) + len(line) + 1 > config.CHUNK_MAX_CHARS and len(cur) > len(header):
            pieces.append("\n".join(cur))
            cur = list(header)
        cur.append(line)
    if len(cur) > len(header):
        pieces.append("\n".join(cur))
    return pieces or [block]


def _split_long(block: str) -> list[str]:
    max_c, overlap = config.CHUNK_MAX_CHARS, config.CHUNK_OVERLAP_CHARS
    if len(block) <= max_c:
        return [block]
    if block.lstrip().startswith("|"):
        return _split_table(block)
    lim = max(max_c - overlap - 1, 50)
    sentences = [s[i : i + lim] for s in re.split(r"(?<=[.!?؟])\s+|\n+", block) for i in range(0, max(len(s), 1), lim)]
    pieces, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > max_c:
            pieces.append(cur)
            tail = cur[-overlap:]
            tail = tail[tail.find(" ") + 1 :] if " " in tail else ""
            cur = f"{tail} {s}".strip()
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def chunk_pages(pages: list[tuple[int, str]], source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur: list[tuple[int, str]] = []
    cur_len = 0
    section = cur_section = ""

    def flush() -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        text = "\n\n".join(t for _, t in cur)
        if len(text) >= 20:
            chunks.append(Chunk(text, source, cur[0][0], cur[-1][0], cur_section, len(chunks)))
        cur, cur_len = [], 0

    for page, text in pages:
        if len(_DOT_LEADERS.findall(text)) >= 5:
            continue
        for block in (b.strip() for b in re.split(r"\n\s*\n", _DOT_LEADERS.sub(" … ", text))):
            if not block:
                continue
            if _is_heading(block):
                if cur_len >= config.CHUNK_MIN_CHARS:
                    flush()
                section = _heading_label(block)
            for piece in _split_long(block):
                if cur and cur_len + len(piece) + 2 > config.CHUNK_MAX_CHARS:
                    flush()
                if not cur:
                    cur_section = section
                cur.append((page, piece))
                cur_len += len(piece) + 2
    flush()
    return chunks
