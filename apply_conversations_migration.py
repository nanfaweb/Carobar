"""
Apply the conversations table migration to Supabase.
"""
import os, sys

def load_env(path):
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip()
    return env

SQL = """
CREATE TABLE IF NOT EXISTS conversations (
  id         uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id    uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
  messages   jsonb NOT NULL DEFAULT '[]',
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE tablename='conversations' AND policyname='Users can manage their own conversations'
  ) THEN
    CREATE POLICY "Users can manage their own conversations"
      ON conversations FOR ALL
      USING (auth.uid() = user_id)
      WITH CHECK (auth.uid() = user_id);
  END IF;
END $$;
"""

def main():
    env = load_env('web/.env.local')
    url = env.get('NEXT_PUBLIC_SUPABASE_URL', '')
    service_key = env.get('SUPABASE_SERVICE_ROLE_KEY', '')

    if not url or not service_key:
        print("ERROR: Missing NEXT_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in web/.env.local")
        sys.exit(1)

    try:
        import psycopg2
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'psycopg2-binary'])
        import psycopg2

    project_ref = url.replace('https://', '').split('.')[0]
    pg_url = (
        f"postgresql://postgres.{project_ref}:{service_key}"
        f"@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres"
    )

    print(f"Connecting to Supabase project: {project_ref}")
    try:
        conn = psycopg2.connect(pg_url)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(SQL)
        cur.close()
        conn.close()
        print("[OK] conversations table is ready!")
    except Exception as e:
        print(f"[ERROR] {e}")
        print("\nFallback — run this SQL manually in the Supabase SQL Editor:")
        print(SQL)

if __name__ == '__main__':
    main()
