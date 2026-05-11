import os
from dotenv import load_dotenv
load_dotenv('web/.env.local')
from supabase import create_client
from sentence_transformers import SentenceTransformer

def main():
    url = os.environ.get('SUPABASE_URL') or os.environ.get('NEXT_PUBLIC_SUPABASE_URL')
    key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY') or os.environ.get('NEXT_PUBLIC_SUPABASE_ANON_KEY')
    client = create_client(url, key)

    
    print("Loading model...")
    model = SentenceTransformer("all-mpnet-base-v2")
    
    print("Fetching listings...")
    res = client.table('listings').select('listing_id, title, location, price_display, mileage_km, fuel_type, transmission, engine_cc, year, embedding').execute()
    
    records = res.data
    if not records:
        print("No records found.")
        return
        
    print(f"Found {len(records)} records. Generating embeddings for those missing them...")
    
    updates = 0
    for r in records:
        if r['embedding'] is not None:
            continue
            
        search_text = (
            f"{r['year']} {r['title']} for sale in {r['location']}. "
            f"Price: {r['price_display']}. Mileage: {r['mileage_km']}. "
            f"Fuel: {r['fuel_type']}. Transmission: {r['transmission']}. "
            f"Engine: {r['engine_cc']}."
        )
        
        emb = model.encode(search_text).tolist()
        client.table('listings').update({'embedding': emb}).eq('listing_id', r['listing_id']).execute()
        updates += 1
        print(f"Updated {updates} embeddings...", end='\r')
        
    print(f"\nDone! Updated {updates} embeddings.")

if __name__ == '__main__':
    main()
