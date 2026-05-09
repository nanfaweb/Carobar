"""
apply_migration.py — Creates the `favorites` table in your live Supabase DB.

Usage:
    python apply_migration.py

Add your service role key to web/.env.local first:
    SUPABASE_SERVICE_ROLE_KEY=eyJhbGc...
Find it at: Supabase Dashboard → Project Settings → API → service_role secret
"""

import os, sys, json

def load_env(path: str) -> dict:
    env: dict[str, str] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    return env

MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS favorites (
  id         uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id    uuid REFERENCES auth.users(id)         ON DELETE CASCADE NOT NULL,
  listing_id BIGINT REFERENCES listings(listing_id) ON DELETE CASCADE NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, listing_id)
);

ALTER TABLE favorites ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='favorites' AND policyname='Users can view their own favorites') THEN
    CREATE POLICY "Users can view their own favorites"   ON favorites FOR SELECT USING (auth.uid() = user_id);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='favorites' AND policyname='Users can insert their own favorites') THEN
    CREATE POLICY "Users can insert their own favorites" ON favorites FOR INSERT WITH CHECK (auth.uid() = user_id);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='favorites' AND policyname='Users can delete their own favorites') THEN
    CREATE POLICY "Users can delete their own favorites" ON favorites FOR DELETE USING (auth.uid() = user_id);
  END IF;
END $$;
"""

def main():
    env = load_env("web/.env.local")
    url = env.get("NEXT_PUBLIC_SUPABASE_URL", "")
    service_key = env.get("SUPABASE_SERVICE_ROLE_KEY", "")

    if not url:
        print("ERROR: NEXT_PUBLIC_SUPABASE_URL missing from web/.env.local"); sys.exit(1)
    if not service_key or service_key == "YOUR_SERVICE_ROLE_KEY_HERE":
        print("ERROR: Add your SUPABASE_SERVICE_ROLE_KEY to web/.env.local")
        print("       Supabase Dashboard → Project Settings → API → service_role")
        sys.exit(1)

    try:
        import httpx
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx"])
        import httpx

    project_ref = url.replace("https://", "").split(".")[0]
    pg_url = f"postgresql://postgres.{project_ref}:{service_key}@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"

    print(f"Connecting to Supabase project: {project_ref}")
    try:
        import psycopg2
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary"])
        import psycopg2

    try:
        conn = psycopg2.connect(pg_url)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(MIGRATION_SQL)
        cur.close()
        conn.close()
        print("✅  Migration applied! favorites table is ready in Supabase.")
    except psycopg2.OperationalError as e:
        print(f"❌  Could not connect via psycopg2: {e}")
        print("\nFallback: Run this SQL in Supabase Dashboard → SQL Editor:\n")
        print(MIGRATION_SQL)

if __name__ == "__main__":
    main()
