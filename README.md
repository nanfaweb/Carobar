# Carobar

Used-car search for the Pakistani market. Next.js frontend, FastAPI RAG backend, and a PakWheels scrape → clean → store → embed pipeline backed by Supabase.

Chat in natural language, get listing matches with photos, run a fair-price analysis, and receive profile-based recommendations.

## Features

- **RAG chat** — Groq (Llama 3.3) answers grounded in stored listings; streams status as it searches
- **On-demand scrape** — if the DB has too few matches for a make, the backend runs the pipeline and retries
- **Price analysis** — verdict (Great Deal / Fair Price / Overpriced) plus a WhatsApp-style negotiation draft
- **Recommendations** — matches from budget, commute, family size, fuel, and transmission
- **Auth, favorites, profiles** — Supabase Auth, saved listings, and user profiles
- **Pipeline** — scrape PakWheels → clean → store in Supabase → sentence-transformer embeddings (`all-mpnet-base-v2`)

## Stack

| Layer | Tech |
| --- | --- |
| Web | Next.js 16, React 19, TypeScript, Tailwind CSS 4 |
| API | FastAPI, Uvicorn, Groq, sentence-transformers |
| Data | Supabase (Postgres + RPC `match_listings`) |
| Ingest | Python scraper (`requests` + BeautifulSoup) |

## Layout

```
Carobar/
├── web/                 # Next.js app (chat, favorites, login, profile)
├── backend/             # FastAPI RAG API (main.py)
├── scraper.py           # PakWheels used-car listings
├── cleaner.py
├── storer.py
├── update_embeddings.py
├── run_pipeline.py      # scrape → clean → store → embed
└── apply_*.py           # Supabase migrations / RLS helpers
```

## Setup

### 1. Environment

Create `web/.env.local`:

```env
NEXT_PUBLIC_SUPABASE_URL=your-supabase-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
GROQ_API_KEY=your-groq-key
```

The FastAPI backend also loads this file.

### 2. Frontend

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### 3. Backend

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

On Windows you can run `start_backend.bat` instead.

Health check: `GET http://localhost:8000/health`

### 4. Ingest listings (optional)

```bash
python run_pipeline.py --pages 2 --make toyota --city lahore
```

Or scrape only:

```bash
python scraper.py --pages 5 --city lahore --make honda
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness |
| POST | `/chat` | Streaming RAG chat (NDJSON) |
| POST | `/analyze-price` | Fair-price verdict + negotiation text |
| POST | `/recommendations` | Profile-based listing matches |

CORS is open to `http://localhost:3000`.
