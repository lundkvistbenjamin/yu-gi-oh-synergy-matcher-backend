import logging
import numpy as np
from fastapi import APIRouter, HTTPException
from app.config import MONSTER_TYPES
from app.services.model_loader import ModelService
from app.schemas.predict import CardPredictionRequest, PredictionResponse

logger = logging.getLogger("uvicorn.error")
router = APIRouter(prefix="/api")

@router.get("/health")
def health_check():
    return {"status": "healthy"}

@router.get("/metadata")
def get_metadata():
    try:
        _, encoders, _ = ModelService.load_resources()
    except Exception as e:
        logger.error(f"[DEBUG LOG] Metadata service error: {str(e)}")
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
        "monster_types": MONSTER_TYPES
    }

@router.post("/predict", response_model=PredictionResponse)
async def predict(payload: CardPredictionRequest):
    try:
        model, encoders, target_encoder = ModelService.load_resources()
    except Exception as e:
        logger.error(f"[DEBUG LOG] Prediction resource fetch failure: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction engine configuration error.")

    # Extract, clean, and validate features using fast dictionary assignments
    processed_features = {}
    for col in ['type', 'race', 'attribute']:
        val = getattr(payload, col).strip()
        le = encoders[col]
        
        if val in le.classes_:
            processed_features[col] = le.transform([val])[0]
        else:
            logger.warning(f"[DEBUG LOG] Anomaly validation break. Column: {col}, Value provided: {val}")
            raise HTTPException(
                status_code=400, 
                detail="Malformed parameters: Attribute mapping anomaly."
            )

    # Compile data vector aligned explicitly to match structural training columns:
    # ['type', 'race', 'atk', 'def', 'level', 'attribute']
    feature_vector = np.array([[
        processed_features['type'],
        processed_features['race'],
        payload.atk,
        payload.get_defense(),
        payload.level,
        processed_features['attribute']
    ]])

    try:
        # Obtain prediction probabilities for all classes
        probabilities = model.predict_proba(feature_vector)[0]
        
        # Get top 3 predicted class indices sorted by probability descending
        top_3_indices = np.argsort(probabilities)[::-1][:3]
        
        # Decode top predictions and form response structure
        top_predictions = []
        for idx in top_3_indices:
            archetype_name = target_encoder.inverse_transform([idx])[0]
            confidence = round(float(probabilities[idx]) * 100, 2)
            top_predictions.append({
                "archetype": str(archetype_name),
                "confidence": confidence
            })

        top_prediction = top_predictions[0]["archetype"]

        return {
            "prediction": str(top_prediction),
            "top_predictions": top_predictions
        }
    except Exception as e:
        logger.error(f"[DEBUG LOG] Model prediction step exception: {str(e)}")
        raise HTTPException(status_code=500, detail="Algorithmic parsing exception.")