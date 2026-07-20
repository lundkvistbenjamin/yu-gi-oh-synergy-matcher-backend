from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import numpy as np

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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
            # Print the actual traceback out to the Vercel logging window
            print(f"[DEBUG LOG] Resource loading failed: {str(e)}")
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
    except Exception as e:
        print(f"[DEBUG LOG] Metadata service error: {str(e)}")
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
    except Exception as e:
        print(f"[DEBUG LOG] Prediction resource fetch failure: {str(e)}")
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

    # Extract, clean, and validate features using fast dictionary assignments
    processed_features = {}
    for col in ['type', 'race', 'attribute']:
        val = str(stats.get(col, '')).strip()
        le = encoders[col]
        
        if val in le.classes_:
            processed_features[col] = le.transform([val])[0]
        else:
            print(f"[DEBUG LOG] Anomaly validation break. Column: {col}, Value provided: {val}")
            raise HTTPException(
                status_code=400, 
                detail=f"Malformed parameters: Attribute mapping anomaly."
            )

    # Compile data vector aligned explicitly to match structural training columns:
    # ['type', 'race', 'atk', 'def', 'level', 'attribute']
    feature_vector = np.array([[
        processed_features['type'],
        processed_features['race'],
        safe_int('atk'),
        safe_int('def') if 'def' in stats else safe_int('defense'),
        safe_int('level'),
        processed_features['attribute']
    ]])

    try:
        prediction_idx = model.predict(feature_vector)[0]
        archetype = target_encoder.inverse_transform([prediction_idx])[0]
        return {"prediction": str(archetype)}
    except Exception as e:
        print(f"[DEBUG LOG] Model prediction step exception: {str(e)}")
        raise HTTPException(status_code=500, detail="Algorithmic parsing exception.")