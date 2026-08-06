import os
from pathlib import Path

# Vercel-friendly absolute path resolution
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

ALLOWED_ORIGINS = [
    "https://duelistsynergy.vercel.app",
    "http://localhost:3000"
]

MONSTER_TYPES = [
    "Effect Monster", "Normal Monster", "Fusion Monster", "Synchro Monster", 
    "XYZ Monster", "Link Monster", "Pendulum Effect Monster", 
    "Union Effect Monster", "Tuner Monster"
]