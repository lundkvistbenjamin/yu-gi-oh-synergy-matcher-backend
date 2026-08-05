import logging
import joblib
from app.config import MODELS_DIR

logger = logging.getLogger("uvicorn.error")

class ModelService:
    _cache = {}

    @classmethod
    def load_resources(cls):
        # Lazy-load cache to optimize serverless cold starts
        if not cls._cache:
            try:
                cls._cache['model'] = joblib.load(MODELS_DIR / "archetype_model.joblib")
                cls._cache['encoders'] = joblib.load(MODELS_DIR / "label_encoders.joblib")
                cls._cache['target_encoder'] = joblib.load(MODELS_DIR / "target_encoder.joblib")
            except Exception as e:
                # Print the actual traceback out to the Vercel logging window
                logger.error(f"[DEBUG LOG] Resource loading failed: {str(e)}")
                # Prevent exposing internal system directory strings to the client in production
                raise RuntimeError("Backend serialization engines failed to initialize.")
                
        return cls._cache['model'], cls._cache['encoders'], cls._cache['target_encoder']