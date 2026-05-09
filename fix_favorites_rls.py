"""
fix_favorites_rls.py — Creates favorites table + RLS policies via Supabase Management API.
"""
import os, sys

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

SQL = """
CREATE TABLE IF NOT EXISTS favorites (
  id         uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id    uuid REFERENCES auth.users(id)         ON DELETE CASCADE NOT NULL,
  listing_id BIGINT REFERENCES listings(listing_id) ON DELETE CASCADE NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, listing_id)
);

ALTER TABLE favorites ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view their own favorites"   ON favorites;
DROP POLICY IF EXISTS "Users can insert their own favorites" ON favorites;
DROP POLICY IF EXISTS "Users can delete their own favorites" ON favorites;

CREATE POLICY "Users can view their own favorites"
  ON favorites FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own favorites"
  ON favorites FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own favorites"
  ON favorites FOR DELETE USING (auth.uid() = user_id);
"""

def main():
    try:
        import httpx
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx"])
        import httpx

    env = load_env("web/.env.local")
    url = env.get("NEXT_PUBLIC_SUPABASE_URL", "")
    service_key = env.get("SUPABASE_SERVICE_ROLE_KEY", "")

    if not url or not service_key:
        print("ERROR: Missing NEXT_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
        sys.exit(1)

    project_ref = url.replace("https://", "").split(".")[0]
    print(f"Project: {project_ref}")

    # Try Supabase Management API
    mgmt_url = f"https://api.supabase.com/v1/projects/{project_ref}/database/query"
    headers = {
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
    }

    print("Trying Supabase Management API...")
    try:
        resp = httpx.post(mgmt_url, json={"query": SQL}, headers=headers, timeout=30)
        print(f"Status: {resp.status_code}")
        if resp.status_code in (200, 201):
            print("SUCCESS: favorites table and RLS policies created!")
            return
        else:
            print(f"Response: {resp.text[:300]}")
    except Exception as e:
        print(f"Management API failed: {e}")

    # Fallback: try the direct DB REST endpoint using PostgREST with service role
    print("\nTrying via PostgREST exec function...")
    rest_url = f"{url}/rest/v1/rpc/exec"
    try:
        resp2 = httpx.post(
            rest_url,
            json={"sql": SQL},
            headers={
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        if resp2.status_code in (200, 201):
            print("SUCCESS via PostgREST exec!")
            return
        else:
            print(f"PostgREST exec status: {resp2.status_code} — {resp2.text[:200]}")
    except Exception as e:
        print(f"PostgREST exec failed: {e}")

    # Last resort — print the SQL for manual execution
    print("\n" + "=" * 60)
    print("MANUAL ACTION REQUIRED")
    print(f"Go to: https://supabase.com/dashboard/project/{project_ref}/sql/new")
    print("Paste and run the SQL below:")
    print("=" * 60)
    print(SQL)

if __name__ == "__main__":
    main()
