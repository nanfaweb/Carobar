-- ============================================================
-- CAROBAR — User Profiles Migration
-- Run in Supabase SQL Editor (Dashboard → SQL Editor → New Query)
-- ============================================================

CREATE TABLE IF NOT EXISTS user_profiles (
  user_id                uuid REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
  budget_max             BIGINT       NOT NULL DEFAULT 3000000,
  commute_km             INT          NOT NULL DEFAULT 20,
  family_size            INT          NOT NULL DEFAULT 2,
  fuel_preference        TEXT         NOT NULL DEFAULT 'Any',
  transmission_preference TEXT        NOT NULL DEFAULT 'Any',
  priorities             JSONB        NOT NULL DEFAULT '[]',
  updated_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Users can view their own profile"
  ON user_profiles FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own profile"
  ON user_profiles FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own profile"
  ON user_profiles FOR UPDATE
  USING (auth.uid() = user_id);
