# Yu-Gi-Oh! Synergy Matcher Engine

A high-performance machine learning backend and API service engineered to ingest card datasets, train random forest classification models, and serve real-time archetype compatibility predictions.

## Overview

The backend architecture manages dataset transformations and model inference workflows by:
* Ingesting raw card JSON records from the YGOPRODeck public API
* Sanitizing missing values, encoding categorical text fields, and compiling clean feature vectors
* Training compressed Random Forest classifiers optimized for low-memory serverless deployments
* Exposing CORS-restricted REST endpoints to process incoming prediction queries asynchronously

## Features

### Data Pipeline and Feature Engineering
* Automated ingestion routines filtering out non-archetype card entries
* Vector cleaning that imputes missing numerical statistics and normalizes categorical features
* Scikit-learn LabelEncoder serialization ensuring deterministic feature mapping between pipeline runs

### Machine Learning and Optimization
* Memory-constrained Random Forest configuration using strict depth caps and compressed serialization (`joblib` with high compression)
* Lazy-loading resource cache within FastAPI contexts to minimize serverless cold starts
* Inverse target decoding returning human-readable archetype classifications from internal predictions

### Security and API Protection
* Input sanitization enforcing numerical boundaries (0-99999) to protect against parameter overflow attacks
* Strict CORS policy restricting cross-origin requests exclusively to the production frontend domain
* Explicit error handling that suppresses internal system stack traces from client-facing HTTP responses
* Hardened production configuration disabling public Swagger and ReDoc documentation routes

## System Architecture

The service operates across an isolated three-phase ML pipeline:

1. **Ingestion & Data Cleansing (`pipeline/`):** Fetches card datasets, filters records with missing archetypes, cleans null fields, and exports structured CSV tabular data.
2. **Model Training (`pipeline/train_model.py`):** Encodes text features, fits a balanced Random Forest Classifier, and serializes compiled model artifacts and encodings to binary assets.
3. **Inference API Layer (`main.py`):** Runs a FastAPI server that receives incoming card parameters, constructs feature arrays dynamically, and predicts matching archetypes in real time.

## Technical Details

### Resource Management Strategy
To operate within tight serverless memory limits, model components are loaded lazily upon the first incoming request rather than during initial script evaluation. Serialized joblib artifacts utilize `compress=3` to reduce footprint sizes, ensuring rapid cold-start initialization and optimal execution throughput.

### Dependencies and Tech Stack
* **Runtime Environment:** Python 3.11+
* **Web Framework:** FastAPI / Uvicorn
* **Machine Learning Engine:** `scikit-learn`, `pandas`, `numpy`
* **Model Serialization:** `joblib`
* **Data Ingestion:** `requests`
* **Deployment Target:** Vercel Serverless Functions

## Output Interpretation

* **Health Endpoint (`/api/health`):** Returns service status checks for runtime monitoring.
* **Metadata Endpoint (`/api/metadata`):** Returns valid categories for dropdown population and feature alignment.
* **Prediction Endpoint (`/api/predict`):** Evaluates input feature vectors and returns the predicted archetype string.

## Limitations

* Model training parameters are strictly constrained to remain within Vercel's serverless RAM memory boundaries.
* Prediction accuracy is inherently bounded by the subset of cards with explicitly labeled archetypes in the YGOPRODeck database.
* Data updates require executing the offline pipeline scripts and re-deploying updated model artifacts.

## Security Note

Service endpoint routes enforce strict input boundary checks and CORS verification. System exceptions during dataset processing log internally for debugging while returning generic status codes to prevent environment exposure.

## License

MIT License - see [LICENSE](LICENSE) file for details.