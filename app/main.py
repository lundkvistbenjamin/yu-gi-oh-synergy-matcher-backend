from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import joblib
import os
import pandas as pd

# 1. Initialize the Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 2. Automatic CORS configuration (FastAPI handles OPTIONS requests automatically)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Vercel-friendly absolute pathing
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Lazy load cache to optimize serverless cold starts on Vercel
MODEL_CACHE = {}

def load_resources():
    if not MODEL_CACHE:
        try:
            MODEL_CACHE['model'] = joblib.load(os.path.join(MODELS_DIR, "archetype_model.joblib"))
            MODEL_CACHE['encoders'] = joblib.load(os.path.join(MODELS_DIR, "label_encoders.joblib"))
            MODEL_CACHE['target_encoder'] = joblib.load(os.path.join(MODELS_DIR, "target_encoder.joblib"))
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")
    return MODEL_CACHE['model'], MODEL_CACHE['encoders'], MODEL_CACHE['target_encoder']


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


@app.get("/api/metadata")
def get_metadata():
    _, encoders, _ = load_resources()
    
    def clean_labels(encoder):
        return [
            str(label) for label in encoder.classes_ 
            if label is not None and str(label).lower() != 'nan' and str(label).upper() != 'NONE'
        ]

    return {
        "types": clean_labels(encoders['type']),
        "races": clean_labels(encoders['race']),
        "attributes": clean_labels(encoders['attribute']),
        "monster_types": [
            "Effect Monster", "Normal Monster", "Fusion Monster", "Synchro Monster", 
            "XYZ Monster", "Link Monster", "Pendulum Effect Monster", 
            "Union Effect Monster", "Tuner Monster"
        ]
    }


@app.post("/api/predict")
@limiter.limit("10/minute")
async def predict(request: Request, stats: dict):
    model, encoders, target_encoder = load_resources()

    # Safely extract and parse numerical inputs, defaulting to -1
    def safe_int(key):
        val = stats.get(key)
        if val is None or str(val).strip() == "":
            return -1
        try:
            parsed = int(float(val))
            return parsed if parsed >= 0 else -1
        except ValueError:
            return -1

    # Format input data structurally matching the training dataset
    input_df = pd.DataFrame({
        'type': [stats.get('type', '')],
        'race': [stats.get('race', '')],
        'atk': [safe_int('atk')],
        'def': [safe_int('def') if 'def' in stats else safe_int('defense')],
        'level': [safe_int('level')],
        'attribute': [stats.get('attribute', '')]
    })
    
    # Encode categorical columns securely
    for col in ['type', 'race', 'attribute']:
        le = encoders[col]
        val = input_df[col].iloc[0]
        
        if val in le.classes_:
            input_df[col] = le.transform([val])[0]
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Value '{val}' for field '{col}' was not found in the training dataset."
            )

    # Make and reverse-transform prediction
    prediction_idx = model.predict(input_df)[0]
    archetype = target_encoder.inverse_transform([prediction_idx])[0]
    
    return {"prediction": archetype}