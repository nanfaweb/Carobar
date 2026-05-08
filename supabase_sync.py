import os
import re
import json
import pandas as pd
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datetime import datetime

# --- CONFIGURATION ---
CSV_PATH = "pakwheels-dataset/PakWheels Dataset.csv"
ENV_PATH = "web/.env.local"
MODEL_NAME = "all-mpnet-base-v2"  # 768 dimensions
BATCH_SIZE = 100

def load_env(path):
    env_vars = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    env_vars[key] = value
    return env_vars

def parse_price(price_str):
    if not isinstance(price_str, str) or not price_str:
        return 0
    s = price_str.strip().lower()
    try:
        # Remove "PKR", commas, etc.
        number_str = re.sub(r"[^\d.]", "", s.replace(",", ""))
        if not number_str:
            return 0
        number = float(number_str)
        if "lakh" in s:
            return int(number * 100_000)
        elif "crore" in s:
            return int(number * 10_000_000)
        return int(number)
    except Exception:
        return 0

def parse_mileage(mileage_str):
    if not isinstance(mileage_str, str) or not mileage_str:
        return 0
    try:
        return int(re.sub(r"[^\d]", "", mileage_str))
    except Exception:
        return 0

def parse_engine(engine_str):
    if not isinstance(engine_str, str) or not engine_str:
        return 0
    try:
        return int(re.sub(r"[^\d]", "", engine_str))
    except Exception:
        return 0

def sync():
    # 1. Load Environment Variables
    env = load_env(ENV_PATH)
    url = env.get("NEXT_PUBLIC_SUPABASE_URL")
    key = env.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    
    if not url or not key:
        print("Error: Supabase URL or Key not found in web/.env.local")
        return

    # 2. Initialize Supabase & Embedding Model
    print(f"Connecting to Supabase at {url}...")
    supabase: Client = create_client(url, key)
    
    print(f"Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    # 3. Load Data
    print(f"Reading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Map CSV columns to Database columns
    # Header: Name,Price,Year,Millage,Fuel,Transmission,Province,Color,Assembly,Engine Capacity,Body Type,Ad Reference,Features,Owner Name,url
    
    records = []
    print("Cleaning and preparing data...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            title = str(row['Name'])
            # Extract make (first word)
            make = title.split(' ')[0]
            # Model is roughly the rest
            model_name = ' '.join(title.split(' ')[1:])
            
            price_pkr = parse_price(row['Price'])
            mileage_km = parse_mileage(row['Millage'])
            engine_cc = parse_engine(row['Engine Capacity'])
            
            # Construct text for RAG embedding
            # This is what the AI will "search" against
            search_text = (
                f"{row['Year']} {title} for sale in {row['Province']}. "
                f"Price: {row['Price']}. Mileage: {row['Millage']}. "
                f"Fuel: {row['Fuel']}. Transmission: {row['Transmission']}. "
                f"Engine: {row['Engine Capacity']}. Features: {row['Features']}."
            )
            
            record = {
                "listing_id": int(row['Ad Reference']),
                "title": title,
                "make": make,
                "model": model_name,
                "year": int(row['Year']),
                "engine_cc": engine_cc,
                "price_pkr": price_pkr,
                "price_display": str(row['Price']),
                "mileage_km": mileage_km,
                "fuel_type": str(row['Fuel']),
                "transmission": str(row['Transmission']),
                "location": str(row['Province']),
                "listing_url": str(row['url']),
                "scraped_at": datetime.now().isoformat(),
                "_search_text": search_text # Temporary field for embedding
            }
            records.append(record)
        except Exception as e:
            continue

    # 4. Process Embeddings and Upload in Batches
    print(f"Starting upload of {len(records)} records in batches of {BATCH_SIZE}...")
    
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        
        # Generate embeddings for the batch
        texts = [r["_search_text"] for r in batch]
        embeddings = model.encode(texts)
        
        # Add embeddings to records and remove the temporary search text
        for record, embedding in zip(batch, embeddings):
            record["embedding"] = embedding.tolist()
            del record["_search_text"]
        
        # Upload to Supabase
        try:
            # Upsert handles duplicates if Ad Reference exists
            response = supabase.table("listings").upsert(batch).execute()
            if i % (BATCH_SIZE * 5) == 0:
                print(f"Uploaded {i + len(batch)} / {len(records)}...")
        except Exception as e:
            print(f"Error uploading batch at index {i}: {e}")

    print("Sync complete!")

if __name__ == "__main__":
    sync()
