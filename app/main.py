from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import pandas as pd

app = FastAPI(
    title="Duelist Synergy API",
    docs_url=None, # Disables automatic Swagger UI documentation for production security
    redoc_url=None # Disables ReDoc documentation
)

# 1. CORS Configuration
ALLOWED_ORIGINS = [
    "https://yu-gi-oh-synergy-matcher-frontend.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET"], # Restrict to only the methods your app actually uses
    allow_headers=["Content-Type", "Authorization"],
)

# Vercel-friendly absolute path mapping
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Lazy-load cache to optimize serverless cold starts
MODEL_CACHE = {}

def load_resources():
    if not MODEL_CACHE:
        try:
            MODEL_CACHE['model'] = joblib.load(os.path.join(MODELS_DIR, "archetype_model.joblib"))
            MODEL_CACHE['encoders'] = joblib.load(os.path.join(MODELS_DIR, "label_encoders.joblib"))
            MODEL_CACHE['target_encoder'] = joblib.load(os.path.join(MODELS_DIR, "target_encoder.joblib"))
        except Exception as e:
            # Prevent exposing internal system directory strings to the client in production
            raise RuntimeError("Backend serialization engines failed to initialize.")
    return MODEL_CACHE['model'], MODEL_CACHE['encoders'], MODEL_CACHE['target_encoder']


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


@app.get("/api/metadata")
def get_metadata():
    try:
        _, encoders, _ = load_resources()
    except Exception:
        raise HTTPException(status_code=500, detail="Metadata service currently unavailable.")
    
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
async def predict(stats: dict):
    # Robust input validation protecting the ML pipeline from malicious payloads
    if not stats or not isinstance(stats, dict):
        raise HTTPException(status_code=400, detail="Invalid request payload structure.")

    try:
        model, encoders, target_encoder = load_resources()
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction engine configuration error.")

    # Explicit input scrubbing and bounds protection
    def safe_int(key):
        val = stats.get(key)
        if val is None or str(val).strip() == "":
            return -1
        try:
            parsed = int(float(val))
            # Enforce reasonable numeric boundaries to prevent overflow attacks
            return parsed if 0 <= parsed <= 99999 else -1
        except (ValueError, TypeError):
            return -1

    # Format input data structurally matching the training dataset
    input_df = pd.DataFrame({
        'type': [str(stats.get('type', '')).strip()],
        'race': [str(stats.get('race', '')).strip()],
        'atk': [safe_int('atk')],
        'def': [safe_int('def') if 'def' in stats else safe_int('defense')],
        'level': [safe_int('level')],
        'attribute': [str(stats.get('attribute', '')).strip()]
    })
    
    # Secure categorical value confirmation
    for col in ['type', 'race', 'attribute']:
        le = encoders[col]
        val = input_df[col].iloc[0]
        
        if val in le.classes_:
            input_df[col] = le.transform([val])[0]
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Malformed parameters: Attribute mapping anomaly."
            )

    try:
        prediction_idx = model.predict(input_df)[0]
        archetype = target_encoder.inverse_transform([prediction_idx])[0]
        return {"prediction": str(archetype)}
    except Exception:
        raise HTTPException(status_code=500, detail="Algorithmic parsing exception.")