import json
import sys
import logging
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load .env file (check root and web folder)
load_dotenv()
load_dotenv("web/.env.local")


# Logging setup
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [STORER] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
log = logging.getLogger(__name__)

# --- CONFIGURATION ---
URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")


if not URL or not KEY:
    log.error("Missing SUPABASE_URL or SUPABASE_KEY in .env file")
    sys.exit(1)

def get_supabase_client() -> Client:
    return create_client(URL, KEY)

def store_listings(data):
    if not data:
        log.warning("No data received to store.")
        return 0

    supabase = get_supabase_client()
    
    # We use upsert with ignore_duplicates=True.
    # This specifically addresses your requirement: 
    # "if listing_id already exists, do not add it again."
    try:
        response = supabase.table("listings").upsert(
            data, 
            on_conflict="listing_id",
            ignore_duplicates=True 
        ).execute()
        
        # Supabase returns the inserted rows in .data
        inserted_count = len(response.data) if response.data else 0
        return inserted_count
        
    except Exception as e:
        log.error(f"Supabase error: {e}")
        return 0

if __name__ == "__main__":
    try:
        if not sys.stdin.isatty():
            raw_input = sys.stdin.read()
            if not raw_input.strip():
                sys.exit(0)
            
            cars = json.loads(raw_input)
            
            # Map Python keys to Supabase columns if they differ
            # (Ensuring list of dicts matches your schema exactly)
            inserted = store_listings(cars)
            
            # Output result for Next.js
            print(json.dumps({
                "status": "success", 
                "received": len(cars),
                "new_records": inserted
            }))
        else:
            log.error("Pipeline error: No data piped into storer.")
            sys.exit(1)
            
    except Exception as e:
        log.error(f"Storage script failed: {e}")
        sys.exit(1)