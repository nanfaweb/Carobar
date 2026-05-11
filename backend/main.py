"""
Carobar RAG Backend — FastAPI
Endpoints:
  POST /chat           — RAG chat with car listings
  POST /analyze-price  — AI fair-price verdict + negotiation draft
  POST /recommendations — Personalised car matches from user profile

Start:  uvicorn backend.main:app --reload --port 8000
"""

import os
import subprocess
import json
import asyncio
from functools import lru_cache

from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from groq import Groq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv("web/.env.local")

SUPABASE_URL    = os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL      = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-mpnet-base-v2"
MATCH_COUNT     = 5
MATCH_THRESHOLD = 0.20  # Lowered from 0.35 to be more inclusive


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    print("Loading SentenceTransformer …")
    return SentenceTransformer(EMBEDDING_MODEL)

@lru_cache(maxsize=1)
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def get_groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    return Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Carobar RAG API", version="2.0.0")

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
    history: list[dict[str, str]] = []

class ChatResponse(BaseModel):
    reply: str
    cars: list[dict[str, Any]]

class PriceAnalysisRequest(BaseModel):
    listing_id: int
    title: str
    make: str
    model: str
    year: int
    price_pkr: int
    mileage_km: int
    fuel_type: str
    transmission: str
    location: str = ""

class PriceAnalysisResponse(BaseModel):
    verdict: str              # "Great Deal" | "Fair Price" | "Overpriced"
    avg_similar_price: int
    similar_count: int
    price_difference: int
    price_difference_pct: float
    analysis: str
    negotiation_message: str

class UserProfileData(BaseModel):
    budget_max: int
    commute_km: int
    family_size: int
    fuel_preference: str
    transmission_preference: str
    priorities: list[str]

class RecommendationsRequest(BaseModel):
    profile: UserProfileData

class RecommendedCar(BaseModel):
    listing_id: int
    title: str
    make: str
    model: str
    year: int
    price_pkr: int
    price_display: str
    mileage_km: int
    fuel_type: str
    transmission: str
    location: str
    hero_image: str | None
    listing_url: str | None
    similarity: float | None
    why_match: str

class RecommendationsResponse(BaseModel):
    recommendations: list[dict[str, Any]]

# ---------------------------------------------------------------------------
# Pipeline Helpers
# ---------------------------------------------------------------------------

def extract_search_entities(query: str) -> dict:
    """Uses LLM to extract make, model and city from the user's query for targeted scraping."""
    client = get_groq_client()
    prompt = f"""Extract car search entities from this query: "{query}"
    Return JSON only: {{"make": "...", "model": "...", "city": "..."}}.
    Use null if not found.
    Common makes: toyota, honda, suzuki, daihatsu, nissan, hyundai, kia, changan, mg, proton, mercedes, bmw, audi, volkswagen, mitsubishi, isuzu, faw, prince.
    Model examples: corolla, civic, alto, cultus, city, elantra, tucson, sportage, stonic, fortuner, prado.
    Common cities: lahore, karachi, islamabad, rawalpindi, peshawar, faisalabad, multan, quetta."""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": "You are a helpful assistant that extracts structured data. Return only valid JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        print(f"[extract_entities] raw result: {result}")
        return result
    except Exception as e:
        print(f"Extraction error: {e}")
        return {"make": None, "model": None, "city": None}


async def run_background_pipeline(make: str, city: str = None):
    """Triggers the full pipeline and yields pulses to keep the connection alive."""
    # Ensure we use the venv python if available
    python_exe = os.sys.executable
    venv_exe = os.path.join(os.getcwd(), "venv", "Scripts", "python.exe")
    if os.path.exists(venv_exe):
        python_exe = venv_exe
        
    cmd = [python_exe, "run_pipeline.py", "--pages", "2", "--make", make]
    if city:
        cmd.extend(["--city", city])
    
    import time
    start_time = time.time()
    print(f"--- Triggering {make} pipeline with {python_exe} ---")
    try:
        # Don't pipe stdout/stderr so they flow directly to the backend terminal
        process = subprocess.Popen(cmd) 
        
        while process.poll() is None:
            if time.time() - start_time > 100:
                process.kill()
                print("Pipeline timed out and was killed.")
                yield json.dumps({"status": "⚠️ Scraping took too long. Showing partial results..."}) + "\n"
                break
                
            yield json.dumps({"status": f"🛰️ Still scraping {make} listings (check backend terminal for logs)..."}) + "\n"
            await asyncio.sleep(5)
        
        if process.returncode != 0 and process.returncode is not None:
            print(f"Pipeline failed with exit code {process.returncode}")
    except Exception as e:
        print(f"Pipeline trigger error: {e}")





# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def retrieve_listings(query: str, count: int = MATCH_COUNT) -> list[dict]:
    embedder  = get_embedder()
    embedding = embedder.encode(query).tolist()
    result = get_supabase().rpc(
        "match_listings",
        {"query_embedding": embedding, "match_count": count, "match_threshold": MATCH_THRESHOLD},
    ).execute()
    cars = result.data or []
    print(f"[retrieve_listings] query='{query}' → {len(cars)} results (vector search)")
    
    # Fallback: if vector search returns nothing, do a plain text search
    if not cars:
        print(f"[retrieve_listings] Falling back to text search for '{query}'")
        # Extract the first word as the likely make
        make_guess = query.strip().split()[0] if query.strip() else query
        text_result = get_supabase().table("listings") \
            .select("listing_id,title,make,model,year,price_pkr,price_display,mileage_km,fuel_type,transmission,location,hero_image,image_db,listing_url") \
            .ilike("make", f"%{make_guess}%") \
            .limit(count) \
            .execute()
        cars = text_result.data or []
        print(f"[retrieve_listings] Text search for make='{make_guess}' → {len(cars)} results")
    
    if cars:
        print(f"[retrieve_listings] Sample car: {cars[0].get('title')} | hero_image={'YES' if cars[0].get('hero_image') else 'NO'}")
    
    return cars


def build_context(cars: list[dict]) -> str:
    if not cars:
        return "No matching listings found in the database."
    lines = ["Here are the most relevant car listings from the database:\n"]
    for i, c in enumerate(cars, 1):
        image_url = c.get('hero_image') or c.get('image_db') or ""
        lines.append(
            f"{i}. **{c.get('title','N/A')}** ({c.get('year','N/A')})\n"
            f"   - Price: {c.get('price_display','N/A')}\n"
            f"   - Mileage: {c.get('mileage_km',0):,} km\n"
            f"   - Fuel: {c.get('fuel_type','N/A')} | Transmission: {c.get('transmission','N/A')}\n"
            f"   - Location: {c.get('location','N/A')}\n"
            f"   - Photo Available: {'Yes (' + image_url + ')' if image_url else 'No'}\n"
            f"   - Listing ID: {c.get('listing_id')}\n"
        )
    return "\n".join(lines)


def build_system_prompt(context: str) -> str:
    return f"""You are Carobar AI, an expert automotive assistant for the Pakistani car market.
Always ground your answers in the provided listings. Be concise, friendly, and helpful.

IMPORTANT IMAGE RULES:
1. If a listing has a 'Photo Available: Yes (URL)', you MUST mention it and should display it using markdown: ![Car Title](URL).
2. If you see a photo URL, NEVER say you don't have images.
3. Your primary goal is to help users find their dream car from the listings provided below.

Current Context:
{context}
"""



def call_groq(system: str, history: list[dict], user_message: str, max_tokens: int = 1024) -> str:
    client = get_groq_client()
    messages = [{"role": "system", "content": system}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=0.7, max_tokens=max_tokens,
    )
    return response.choices[0].message.content

# ---------------------------------------------------------------------------
# Routes — existing
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    async def event_generator():
        try:
            # 1. Extract entities first to clean up the search query
            yield json.dumps({"status": "🔍 Analyzing your query..."}) + "\n"
            entities = extract_search_entities(req.message)
            make = entities.get("make")
            model = entities.get("model")
            city = entities.get("city")
            
            print(f"[chat] Extracted: make={make}, model={model}, city={city}")
            
            # Build a targeted search query from extracted entities
            search_query = req.message
            if make and make.lower() not in ("null", "none", ""):
                parts = [make]
                if model and model.lower() not in ("null", "none", ""):
                    parts.append(model)
                if city and city.lower() not in ("null", "none", ""):
                    parts.append(city)
                search_query = " ".join(parts)
            
            yield json.dumps({"status": "🔎 Searching in database..."}) + "\n"
            cars = retrieve_listings(search_query)
            
            # 2. Check if results actually match the requested make
            # (vector/text search can return irrelevant results)
            if make and make.lower() not in ("null", "none", ""):
                matching = [c for c in cars if make.lower() in c.get("make", "").lower() or make.lower() in c.get("title", "").lower()]
                print(f"[chat] {len(matching)}/{len(cars)} results match make='{make}'")
                
                # If fewer than 3 results match the make, scrape for fresh data
                if len(matching) < 3:
                    yield json.dumps({"status": f"🛰️ Not enough {make} listings. Scraping fresh data..."}) + "\n"
                    async for pulse in run_background_pipeline(make, city if city and city.lower() not in ("null","none","") else None):
                        yield pulse
                    
                    yield json.dumps({"status": "🔄 Updating search results..."}) + "\n"
                    cars = retrieve_listings(search_query)
                    # Re-filter after scrape
                    matching = [c for c in cars if make.lower() in c.get("make", "").lower() or make.lower() in c.get("title", "").lower()]
                    # If we now have matching cars, use those; otherwise use all
                    cars = matching if matching else cars
                else:
                    cars = matching  # Only show cars that match the requested make



            
            # 3. Generate final AI response
            yield json.dumps({"status": "🤖 AI is thinking..."}) + "\n"
            system = build_system_prompt(build_context(cars))
            reply = call_groq(system, req.history, req.message)
            
            # Send the final data
            yield json.dumps({
                "reply": reply,
                "cars": cars,
                "done": True
            }) + "\n"
        except Exception as exc:
            print(f"Chat Error: {exc}")
            yield json.dumps({"error": str(exc)}) + "\n"


    return StreamingResponse(event_generator(), media_type="application/x-ndjson")



# ---------------------------------------------------------------------------
# Route — Price Analyzer
# ---------------------------------------------------------------------------

@app.post("/analyze-price", response_model=PriceAnalysisResponse)
def analyze_price(req: PriceAnalysisRequest):
    """
    Retrieve similar cars from the vector DB, calculate a market average,
    produce a verdict, and draft a negotiation message via Groq.
    """
    query = f"{req.year} {req.make} {req.model} {req.fuel_type} {req.transmission}"
    similar_cars = retrieve_listings(query, count=8)

    # Exclude the listing itself
    others = [
        c for c in similar_cars
        if c.get("listing_id") != req.listing_id and c.get("price_pkr")
    ]

    if others:
        prices    = [c["price_pkr"] for c in others]
        avg_price = int(sum(prices) / len(prices))
    else:
        avg_price = req.price_pkr

    price_diff     = req.price_pkr - avg_price
    price_diff_pct = round((price_diff / avg_price * 100) if avg_price > 0 else 0, 1)

    if price_diff_pct < -10:
        verdict = "Great Deal"
    elif price_diff_pct > 15:
        verdict = "Overpriced"
    else:
        verdict = "Fair Price"

    similar_lines = "\n".join(
        f"- {c.get('title','N/A')}: PKR {c.get('price_pkr',0):,}, {c.get('mileage_km',0):,} km"
        for c in others[:4]
    ) or "No closely similar listings found in our database."

    diff_label = (
        f"PKR {price_diff:,} above average ({price_diff_pct}% overpriced)"
        if price_diff > 0
        else f"PKR {abs(price_diff):,} below average ({abs(price_diff_pct)}% cheaper)"
    )

    prompt = f"""Analyze this car vs similar listings in Pakistan:

Listing: {req.title}
Price: PKR {req.price_pkr:,} | Year: {req.year} | Mileage: {req.mileage_km:,} km
Fuel: {req.fuel_type} | Transmission: {req.transmission} | Location: {req.location}

Similar cars found:
{similar_lines}

Market average price: PKR {avg_price:,}
Price difference: {diff_label}
Verdict: {verdict}

Write:
1. A 2-sentence analysis explaining the verdict using the data above.
2. A polite, data-backed WhatsApp negotiation message to the seller in English.

Format EXACTLY as:
ANALYSIS: <your 2-sentence analysis>
NEGOTIATION: <negotiation message>"""

    analysis = negotiation = ""
    try:
        client   = get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert Pakistani automotive market analyst. Be concise and data-driven."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=600,
        )
        raw = response.choices[0].message.content or ""
        if "ANALYSIS:" in raw and "NEGOTIATION:" in raw:
            parts       = raw.split("NEGOTIATION:")
            analysis    = parts[0].replace("ANALYSIS:", "").strip()
            negotiation = parts[1].strip()
        else:
            analysis    = raw[:400].strip()
            negotiation = (
                f"Hi! I'm interested in your {req.title} listed at PKR {req.price_pkr:,}. "
                f"I noticed similar {req.year} {req.make} {req.model}s are available around "
                f"PKR {avg_price:,}. Would you consider adjusting the price? I'm a serious buyer. Thank you!"
            )
    except Exception:
        analysis = f"This listing is a {verdict.lower()} compared to similar cars in our database."
        negotiation = (
            f"Hi! I'm interested in your {req.title}. Could you consider a lower price? "
            f"Similar listings are around PKR {avg_price:,}. I'm ready to buy. Thank you!"
        )

    return PriceAnalysisResponse(
        verdict=verdict,
        avg_similar_price=avg_price,
        similar_count=len(others),
        price_difference=price_diff,
        price_difference_pct=price_diff_pct,
        analysis=analysis,
        negotiation_message=negotiation,
    )

# ---------------------------------------------------------------------------
# Route — Smart Recommendations
# ---------------------------------------------------------------------------

@app.post("/recommendations", response_model=RecommendationsResponse)
def get_recommendations(req: RecommendationsRequest):
    """
    Build a natural-language query from the user's lifestyle profile,
    retrieve top matching cars, then generate personalised 'why this fits you'
    explanations via Groq.
    """
    p = req.profile

    fuel  = p.fuel_preference if p.fuel_preference.lower() != "any" else ""
    trans = p.transmission_preference if p.transmission_preference.lower() != "any" else ""
    prios = ", ".join(p.priorities) if p.priorities else "reliable affordable"

    commute_desc = (
        "long-distance fuel efficient highway"  if p.commute_km > 50
        else "moderate commute city"            if p.commute_km > 20
        else "short city driving"
    )
    family_desc = (
        "large family 7-seater SUV"  if p.family_size >= 5
        else "family sedan spacious" if p.family_size >= 3
        else "compact"
    )

    query = f"{fuel} {trans} {family_desc} {commute_desc} {prios} budget PKR {p.budget_max:,}".strip()
    cars  = retrieve_listings(query, count=10)

    # Filter to budget
    affordable = [c for c in cars if c.get("price_pkr", 0) <= p.budget_max]
    top_cars   = (affordable or cars)[:3]

    if not top_cars:
        return RecommendationsResponse(recommendations=[])

    profile_summary = (
        f"- Budget: up to PKR {p.budget_max:,}\n"
        f"- Daily commute: {p.commute_km} km\n"
        f"- Family size: {p.family_size} people\n"
        f"- Fuel preference: {p.fuel_preference}\n"
        f"- Transmission: {p.transmission_preference}\n"
        f"- Priorities: {prios}"
    )
    cars_text = "\n".join(
        f"{i+1}. {c.get('title','N/A')} — PKR {c.get('price_pkr',0):,}, "
        f"{c.get('mileage_km',0):,} km, {c.get('fuel_type','N/A')}, "
        f"{c.get('transmission','N/A')}, {c.get('location','N/A')}"
        for i, c in enumerate(top_cars)
    )

    format_instructions = "\n".join([f"MATCH{i+1}: <reason for car {i+1}>" for i in range(len(top_cars))])
    
    prompt = f"""User Profile:
{profile_summary}

Top recommended cars:
{cars_text}

For each car write 1-2 warm sentences explaining WHY it specifically matches this user's commute, family size, budget and priorities.

Format EXACTLY as:
{format_instructions}"""


    matches: dict[str, str] = {"MATCH1": "", "MATCH2": "", "MATCH3": ""}
    try:
        client   = get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a personal car buying advisor. Give warm, personalised recommendations."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=400,
        )
        raw = response.choices[0].message.content or ""
        for key in matches:
            tag = f"{key}:"
            if tag in raw:
                start = raw.index(tag) + len(tag)
                rest  = raw[start:]
                # find next MATCH tag
                end = len(rest)
                for other in matches:
                    if other != key and f"{other}:" in rest:
                        end = min(end, rest.index(f"{other}:"))
                matches[key] = rest[:end].strip()
    except Exception:
        defaults = [
            "A great fit within your budget and priorities.",
            "Matches your commute and family needs well.",
            "Fits your preferences and lifestyle perfectly.",
        ]
        for i, key in enumerate(matches):
            matches[key] = defaults[i]

    result = [
        {**car, "why_match": matches.get(f"MATCH{i+1}") or "Great match for your profile."}
        for i, car in enumerate(top_cars)
    ]

    return RecommendationsResponse(recommendations=result)
