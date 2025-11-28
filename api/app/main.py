import os
import json
import uuid
import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncpg
import joblib

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@db:5432/noisemap")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rf_noise_classifier.joblib")
# If you don't have a model, inference endpoint will return "unknown"

app = FastAPI(title="NoiseMap API")

class Reading(BaseModel):
    device_id: str
    timestamp: datetime.datetime
    lat: float
    lon: float
    db_level: float
    source_label: Optional[str] = None
    features: Optional[dict] = None
    raw_audio_path: Optional[str] = None

async def get_conn():
    return await asyncpg.connect(DATABASE_URL)

@app.post("/ingest")
async def ingest(reading: Reading):
    conn = await get_conn()
    try:
        geom = f"SRID=4326;POINT({reading.lon} {reading.lat})"
        q = """
        INSERT INTO noise_readings (id, device_id, timestamp, lat, lon, geom, db_level, source_label, features, raw_audio_path)
        VALUES ($1,$2,$3,$4,$5,ST_GeomFromText($6),$7,$8,$9,$10)
        """
        await conn.execute(q, str(uuid.uuid4()), reading.device_id, reading.timestamp, reading.lat, reading.lon,
                           geom, reading.db_level, reading.source_label, json.dumps(reading.features or {}), reading.raw_audio_path)
        return {"status": "ok"}
    finally:
        await conn.close()

@app.get("/readings/recent")
async def recent(limit: int = 1000):
    conn = await get_conn()
    try:
        rows = await conn.fetch("SELECT id, device_id, timestamp, lat, lon, db_level, source_label FROM noise_readings ORDER BY timestamp DESC LIMIT $1", limit)
        res = [dict(r) for r in rows]
        return res
    finally:
        await conn.close()

@app.get("/readings/heatmap")
async def heatmap(hours:int = 168):  # default last 7 days
    conn = await get_conn()
    try:
        q = """
        SELECT round(lat::numeric,4) as lat_r, round(lon::numeric,4) as lon_r, AVG(db_level) as avg_db, COUNT(*) as n
        FROM noise_readings
        WHERE timestamp >= now() - ($1 || ' hours')::interval
        GROUP BY lat_r, lon_r;
        """
        rows = await conn.fetch(q, hours)
        return [dict(r) for r in rows]
    finally:
        await conn.close()

# Simple inference using saved joblib model
try:
    MODEL = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
except Exception:
    MODEL = None

@app.post("/infer/classify")
async def classify(features: dict):
    """
    Expects features like {'mfcc_mean':[...], 'rms':..., 'zcr':...}
    """
    if MODEL is None:
        return {"label": "unknown", "detail": "no model available on server"}
    # Construct feature vector consistent with training
    try:
        x = []
        x.extend(features.get("mfcc_mean", []))
        x.append(features.get("rms", 0.0))
        x.append(features.get("zcr", 0.0))
        pred = MODEL.predict([x])[0]
        return {"label": str(pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
