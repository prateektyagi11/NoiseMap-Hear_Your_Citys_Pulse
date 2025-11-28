# NoiseMap — Local Dev (Docker Compose)

## Prereqs
- Docker & Docker Compose installed
- (Optional) A trained model placed at `api/app/models/rf_noise_classifier.joblib` or run `python ml/train_classifier.py` (prepare CSV first)

## Quick start
1. Copy .env.example -> .env and edit if needed.
2. Build & start:
   ```bash
   docker compose up --build
