from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import json

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
MODEL_JSON_PATH = os.path.join(BASE_DIR, "models", "model_data.json")

# Lazy-load cache to optimize serverless cold starts
MODEL_CACHE = {}

def load_resources():
    if not MODEL_CACHE:
        try:
            with open(MODEL_JSON_PATH, "r") as f:
                MODEL_CACHE['data'] = json.load(f)
        except Exception:
            raise RuntimeError("Backend serialization engines failed to initialize.")
    return MODEL_CACHE['data']


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


@app.get("/api/metadata")
def get_metadata():
    try:
        data = load_resources()
    except Exception:
        raise HTTPException(status_code=500, detail="Metadata service currently unavailable.")
    
    encoders = data["encoders"]
    
    def clean_labels(class_list):
        return [
            str(label) for label in class_list 
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
        data = load_resources()
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

    # Extract strings safely matching the structure expected by our encoders
    input_strings = {
        'type': str(stats.get('type', '')).strip(),
        'race': str(stats.get('race', '')).strip(),
        'attribute': str(stats.get('attribute', '')).strip()
    }
    
    # Secure categorical value confirmation using primitive list lookups
    encoded_features = []
    for col in ['type', 'race']:
        classes = data["encoders"][col]
        val = input_strings[col]
        if val in classes:
            encoded_features.append(classes.index(val))
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Malformed parameters: Attribute mapping anomaly."
            )
            
    # Add our raw continuous numeric features safely
    encoded_features.append(safe_int('atk'))
    encoded_features.append(safe_int('def') if 'def' in stats else safe_int('defense'))
    encoded_features.append(safe_int('level'))
    
    # Add final categorical feature (attribute)
    attr_classes = data["encoders"]["attribute"]
    attr_val = input_strings['attribute']
    if attr_val in attr_classes:
        encoded_features.append(attr_classes.index(attr_val))
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Malformed parameters: Attribute mapping anomaly."
        )

    try:
        # PURE PYTHON MATRIX MATHEMATICS FOR INFRASTRUCTURE PROTECTION
        # Computes dot product across all classes using only plain integers/floats
        coefficients = data["coefficients"]
        intercepts = data["intercept"]
        classes = data["classes"]
        
        best_class_idx = 0
        max_score = float('-inf')
        
        for i in range(len(classes)):
            # Calculate linear combinations: score = dot_product(weights, features) + intercept
            current_score = sum(encoded_features[j] * coefficients[i][j] for j in range(len(encoded_features))) + intercepts[i]
            if current_score > max_score:
                max_score = current_score
                best_class_idx = i
                
        archetype = classes[best_class_idx]
        return {"prediction": str(archetype)}
    except Exception:
        raise HTTPException(status_code=500, detail="Algorithmic parsing exception.")