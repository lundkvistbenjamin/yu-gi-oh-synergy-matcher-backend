import logging
import joblib
from app.config import MODELS_DIR

logger = logging.getLogger("uvicorn.error")

class ModelService:
    _cache = {}

    @classmethod
    def load_resources(cls):
        if not cls._cache:
            try:
                cls._cache['model'] = joblib.load(MODELS_DIR / "archetype_model.joblib")
                cls._cache['encoders'] = joblib.load(MODELS_DIR / "label_encoders.joblib")
                cls._cache['target_encoder'] = joblib.load(MODELS_DIR / "target_encoder.joblib")
                logger.info("Successfully loaded ML model and encoders into memory.")
            except Exception as e:
                logger.error(f"Resource loading failed: {str(e)}")
                raise RuntimeError("Backend serialization engines failed to initialize.")
                
        return cls._cache['model'], cls._cache['encoders'], cls._cache['target_encoder']