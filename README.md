# 🎓 University RAG Bot

A Telegram assistant that answers students' questions about university regulations, schedules, and campus locations — **grounded only in the official documents you upload**, using Retrieval-Augmented Generation (RAG) with Google Gemini and Qdrant.

Students can ask in text or voice (Egyptian Arabic), and the bot replies with a sourced answer and, when relevant, a map location.

## ✨ Features

- 📚 **Document-grounded answers** — no guessing; cites file name and page number.
- 🎤 **Voice questions** — speech-to-text through Gemini.
- 📍 **Campus locations** — sends a Telegram map pin for buildings, halls, gates, etc.
- 💬 **Conversation memory** — understands follow-ups like "what about the second semester?".
- 📄 **Multi-format ingestion** — PDF (with OCR for scans), Word (`.docx`), and structured JSON.
- 🔎 **Hybrid retrieval** — vector search + lexical boost, with batched evidence extraction before the final answer.
- 🛡️ **Admin tools** — upload, list, and delete documents straight from Telegram.
- ⏱️ **Smart waiting message** — estimated response time based on rolling stats.
- ⚙️ **Rate-limit aware** — built-in RPM limiter and retry/fallback for Gemini.

## 🏗️ How it works

```
Student question (text / voice)
        │
        ▼
 Query building (+ chat history)
        │
        ▼
 Qdrant vector search ──► lexical re-rank ──► top-K chunks
        │
        ▼
 Batched "notes" extraction (Gemini, parallel)
        │
        ▼
 Final answer + source + location ids (Gemini)
        │
        ▼
 Telegram reply (+ map pin)
```

## 🧰 Tech stack

Python 3.10+ · [aiogram 3](https://docs.aiogram.dev) · [Google Gemini](https://ai.google.dev) (`google-genai`) · [Qdrant](https://qdrant.tech) · pypdf · python-docx · pytest

## 📁 Project structure

```
app/
├── main.py            # entry point
├── config.py          # env-based settings
├── pipeline.py        # RAG pipeline (retrieve → notes → answer)
├── gemini.py          # Gemini client, rate limiting, retries
├── vector_store.py    # Qdrant wrapper
├── ingest.py          # document ingestion CLI
├── documents.py       # PDF / DOCX parsing & chunking
├── jsondata.py        # JSON knowledge ingestion
├── locations.py       # campus locations
├── conversations.py   # per-user chat memory
├── prompts.py         # prompts
├── waiting.py         # ETA / waiting message
└── handlers/          # Telegram handlers (user + admin)
tests/                 # pytest suite
data/pdfs, data/docs   # put your source documents here (git-ignored)
```

## 🚀 Getting started

### 1. Prerequisites
- Python 3.10+
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- A [Gemini API key](https://aistudio.google.com/apikey)
- A Qdrant instance (local Docker or [Qdrant Cloud](https://cloud.qdrant.io))

### 2. Install
```bash
git clone https://github.com/<WaseemMohamed174>/University-RAG-Bot.git
cd University-RAG-Bot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env             # Windows: copy .env.example .env
```
Fill in `BOT_TOKEN`, `GEMINI_API_KEY`, `ADMIN_ID`, and the Qdrant settings.
> Don't know your Telegram ID? Run the bot once and send `/id`.

### 4. Run Qdrant locally (optional)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 5. Add your documents
Put PDFs in `data/pdfs/` and Word/JSON files in `data/docs/`, then:
```bash
python -m app.ingest           # add --force to re-parse everything
```
Or just send the file to the bot as the admin.

### 6. Start the bot
```bash
python -m app.main
```

## 🤖 Bot commands

| Command | Who | Description |
|---|---|---|
| `/start`, `/help` | Everyone | Welcome & usage tips |
| `/new` | Everyone | Start a fresh conversation |
| `/id` | Everyone | Show your Telegram ID |
| `/files` | Admin | List indexed documents |
| `/delete <name>` | Admin | Remove a document |
| *(send a file)* | Admin | Upload PDF / DOCX / JSON (≤ 20 MB) |

## 🧪 Tests

```bash
pip install pytest
pytest -q
```
Tests use an in-memory Qdrant and mocked Gemini, so no API keys are needed.

## ⚠️ Privacy note

Source documents and conversation logs may contain personal or student data. The `data/` folder and `.env` are git-ignored — never commit them.

## 🗺️ Roadmap

- [ ] Docker / docker-compose setup
- [ ] Webhook deployment mode
- [ ] Admin analytics dashboard

## 📄 License

MIT — see [LICENSE](LICENSE).
