-- ============================================================
-- CAROBAR — Supabase Schema
-- Run this entire file in the Supabase SQL editor
-- ============================================================

-- ------------------------------------------------------------
-- 0. Extensions
-- ------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS vector;


-- ------------------------------------------------------------
-- 1. Listings table
-- ------------------------------------------------------------

CREATE TABLE listings (
  listing_id      BIGINT PRIMARY KEY,
  title           TEXT,
  make            TEXT NOT NULL,
  model           TEXT NOT NULL,
  year            INT NOT NULL,
  engine_cc       INT,
  price_pkr       BIGINT NOT NULL,
  price_display   TEXT,
  mileage_km      INT,
  fuel_type       TEXT,
  transmission    TEXT,
  location        TEXT,
  hero_image      TEXT,
  image_db        JSONB DEFAULT '[]',
  listing_url     TEXT,
  updated_at      TIMESTAMPTZ,
  scraped_at      TIMESTAMPTZ DEFAULT NOW(),
  embedding       vector(768)
);


-- ------------------------------------------------------------
-- 2. Indexes — marketplace filtering
-- ------------------------------------------------------------

CREATE INDEX idx_listings_make         ON listings (make);
CREATE INDEX idx_listings_model        ON listings (model);
CREATE INDEX idx_listings_year         ON listings (year);
CREATE INDEX idx_listings_price        ON listings (price_pkr);
CREATE INDEX idx_listings_location     ON listings (location);
CREATE INDEX idx_listings_fuel         ON listings (fuel_type);
CREATE INDEX idx_listings_transmission ON listings (transmission);

-- Composite index for the most common filter combo
CREATE INDEX idx_listings_make_model_year
  ON listings (make, model, year);


-- ------------------------------------------------------------
-- 3. pgvector index — RAG similarity search
-- Populate embeddings before this index matters
-- ------------------------------------------------------------

CREATE INDEX idx_listings_embedding
  ON listings
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);


-- ------------------------------------------------------------
-- 5. pgvector similarity search function
-- Called by FastAPI RAG endpoint to retrieve top-K listings
-- Usage: SELECT * FROM match_listings(query_embedding, 5, 0.5)
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION match_listings (
  query_embedding  vector(768),
  match_count      INT     DEFAULT 5,
  match_threshold  FLOAT   DEFAULT 0.5
)
RETURNS TABLE (
  listing_id     BIGINT,
  title          TEXT,
  make           TEXT,
  model          TEXT,
  year           INT,
  engine_cc      INT,
  price_pkr      BIGINT,
  price_display  TEXT,
  mileage_km     INT,
  fuel_type      TEXT,
  transmission   TEXT,
  location       TEXT,
  hero_image     TEXT,
  listing_url    TEXT,
  similarity     FLOAT
)
LANGUAGE sql STABLE
AS $$
  SELECT
    listing_id,
    title,
    make,
    model,
    year,
    engine_cc,
    price_pkr,
    price_display,
    mileage_km,
    fuel_type,
    transmission,
    location,
    hero_image,
    listing_url,
    1 - (embedding <=> query_embedding) AS similarity
  FROM listings
  WHERE 1 - (embedding <=> query_embedding) > match_threshold
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
$$;
