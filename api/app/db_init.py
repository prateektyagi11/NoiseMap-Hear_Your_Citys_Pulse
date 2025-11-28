import os
import time
import asyncio
import asyncpg

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@db:5432/noisemap")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS noise_readings (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  device_id TEXT,
  timestamp TIMESTAMPTZ NOT NULL,
  lat DOUBLE PRECISION,
  lon DOUBLE PRECISION,
  geom GEOMETRY(Point, 4326),
  db_level DOUBLE PRECISION,
  sample_rate INTEGER,
  duration_seconds DOUBLE PRECISION,
  processed BOOLEAN DEFAULT FALSE,
  source_label TEXT,
  features JSONB,
  raw_audio_path TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_noise_geom ON noise_readings USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_noise_time ON noise_readings (timestamp);
"""

async def init():
    for i in range(30):
        try:
            conn = await asyncpg.connect(DATABASE_URL)
            await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
            await conn.execute('CREATE EXTENSION IF NOT EXISTS postgis;')
            await conn.execute(CREATE_TABLE_SQL)
            await conn.close()
            print("DB initialized")
            return
        except Exception as e:
            print("Waiting for DB...", str(e))
            time.sleep(2)
    raise RuntimeError("Could not initialize DB")

if __name__ == "__main__":
    asyncio.run(init())
