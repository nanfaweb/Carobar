"""
Carobar RAG Backend — FastAPI
Receives a user chat message, generates an embedding, retrieves top-K
matching car listings from Supabase, then asks Google Gemini to craft a
natural-language reply grounded in those listings.

Start:  uvicorn backend.main:app --reload --port 8000
"""

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv("web/.env.local")

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = "all-mpnet-base-v2"   # MUST match what supabase_sync.py used
MATCH_COUNT = 5
MATCH_THRESHOLD = 0.35                  # cosine-similarity lower bound

# ---------------------------------------------------------------------------
# Lazy singletons (loaded once on first request)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    print("Loading SentenceTransformer …")
    return SentenceTransformer(EMBEDDING_MODEL)

@lru_cache(maxsize=1)
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Carobar RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []   # [{"role": "user"|"assistant", "content": "..."}]

class ChatResponse(BaseModel):
    reply: str
    cars: list[dict[str, Any]]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_context(cars: list[dict]) -> str:
    """Turn retrieved listings into a compact markdown table for the LLM."""
    if not cars:
        return "No matching listings found in the database."

    lines = ["Here are the most relevant car listings from the database:\n"]
    for i, c in enumerate(cars, 1):
        lines.append(
            f"{i}. **{c.get('title', 'N/A')}** ({c.get('year', 'N/A')})\n"
            f"   - Price: {c.get('price_display', 'N/A')}\n"
            f"   - Mileage: {c.get('mileage_km', 'N/A'):,} km\n"
            f"   - Fuel: {c.get('fuel_type', 'N/A')} | Transmission: {c.get('transmission', 'N/A')}\n"
            f"   - Location: {c.get('location', 'N/A')}\n"
            f"   - Listing ID: {c.get('listing_id')}\n"
        )
    return "\n".join(lines)


def build_system_prompt(context: str) -> str:
    return f"""You are Carobar AI, an expert automotive assistant for the Pakistani car market.
You have access to real, live car listings retrieved from the Carobar database.
Always ground your answers in the provided listings.
Be concise, friendly, and helpful. Format prices in PKR.
If the user asks something unrelated to cars, politely redirect them.

{context}
"""


def retrieve_listings(query: str) -> list[dict]:
    """Embed the query and call the pgvector match_listings function."""
    embedder = get_embedder()
    embedding = embedder.encode(query).tolist()

    supabase = get_supabase()
    result = supabase.rpc(
        "match_listings",
        {
            "query_embedding": embedding,
            "match_count": MATCH_COUNT,
            "match_threshold": MATCH_THRESHOLD,
        },
    ).execute()

    return result.data or []


def call_gemini(system: str, history: list[dict], user_message: str) -> str:
    """Call Google Gemini with conversation history."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Build Gemini history format
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
        )
    
    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system
        )
    )
    return response.text

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 1. Retrieve relevant listings
    cars = retrieve_listings(req.message)

    # 2. Build context + system prompt
    context = build_context(cars)
    system = build_system_prompt(context)

    # 3. Call Gemini
    try:
        reply = call_gemini(system, req.history, req.message)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return ChatResponse(reply=reply, cars=cars)
