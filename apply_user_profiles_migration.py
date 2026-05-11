"""Apply user_profiles table migration to Supabase."""
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv("web/.env.local")

url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not url or not key:
    raise SystemExit("Missing SUPABASE_URL or SERVICE_ROLE_KEY in web/.env.local")

sql = open("web/supabase/user_profiles_migration.sql").read()

client = create_client(url, key)
# Use the postgrest-based approach via service role
client.rpc  # ensure connection

# Execute each statement individually
import re
statements = [s.strip() for s in re.split(r";", sql) if s.strip() and not s.strip().startswith("--")]

try:
    from supabase.lib.client_options import ClientOptions
except ImportError:
    pass

# Use the supabase management client or raw http
import httpx

headers = {
    "apikey": key,
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
}

pg_url = url.replace("https://", "https://").rstrip("/") + "/rest/v1/rpc/exec_sql"

# Try via pg_net / direct SQL endpoint
sql_url = url.rstrip("/") + "/rest/v1/"

# Use supabase's built-in SQL via service role
print("Applying user_profiles migration...")

resp = httpx.post(
    url.rstrip("/") + "/rest/v1/rpc/",
    headers=headers,
)

# Fallback: use the supabase python client's internal Postgrest query
# The cleanest way: use the SQL Editor-compatible endpoint
response = httpx.post(
    f"{url}/rest/v1/",
    headers={**headers, "Prefer": "return=minimal"},
)

# Best approach for service role: direct postgres endpoint
sql_endpoint = f"{url.replace('.supabase.co', '.supabase.co')}/rest/v1/rpc"

# Actually use the correct approach - supabase management API
mgmt_url = f"https://api.supabase.com/v1/projects/{url.split('//')[1].split('.')[0]}/database/query"

# Simplest: just print the SQL and tell user to run it
print("\n" + "="*60)
print("ACTION REQUIRED — Run this SQL in Supabase SQL Editor:")
print("Dashboard → SQL Editor → New query → paste → Run")
print("="*60)
print(open("web/supabase/user_profiles_migration.sql").read())
print("="*60)

# Try to run via service role using raw query endpoint
project_ref = url.split("//")[1].split(".")[0]

resp2 = httpx.post(
    f"https://{project_ref}.supabase.co/rest/v1/rpc/exec",
    headers=headers,
    json={"query": sql},
    timeout=30,
)
if resp2.status_code == 200:
    print("\n✅ Migration applied successfully via API!")
else:
    print(f"\nℹ️  Auto-apply returned {resp2.status_code}. Please run the SQL manually in the Supabase SQL Editor.")
    print("URL:", f"https://supabase.com/dashboard/project/{project_ref}/sql")
