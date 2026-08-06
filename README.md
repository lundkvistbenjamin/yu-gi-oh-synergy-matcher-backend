# Yu-Gi-Oh! Synergy Matcher Backend

**Live Demo (Frontend):** [https://duelistsynergy.vercel.app/](https://duelistsynergy.vercel.app/)

**Frontend Repository:** [https://github.com/lundkvistbenjamin/yu-gi-oh-synergy-matcher-frontend](https://github.com/lundkvistbenjamin/yu-gi-oh-synergy-matcher-frontend)

Yu-Gi-Oh! Synergy Matcher Engine is a machine learning backend that predicts the most likely archetype for a Yu-Gi-Oh! monster card based on its attributes and statistics. The project combines a data preparation pipeline, a Random Forest classification model, and a FastAPI server to deliver real-time predictions through a lightweight REST API.

## Core Features

### Automated Data Pipeline

The project includes a complete preprocessing pipeline that downloads card data directly from the YGOPRODeck API, filters unsupported records, cleans missing values, and prepares structured datasets for model training.

### Random Forest Classification

A Scikit-learn Random Forest model is trained on cleaned monster card data to recognize archetype patterns from card attributes such as type, race, attribute, attack, defense, and level. Trained models and encoders are serialized with Joblib for efficient deployment.

### FastAPI Prediction Service

The backend exposes REST endpoints for health checks, metadata retrieval, and archetype prediction. Model resources are loaded lazily to reduce serverless cold-start overhead while maintaining fast inference times.

### Production-Oriented API Design

The API includes strict input validation via Pydantic schemas, CORS protection, controlled error handling, and disabled Swagger documentation in production environments. Predictions return both the highest-confidence archetype and the top three ranked predictions.

## Tech Stack

### Backend

- Python 3.11
- FastAPI
- Uvicorn
- Pydantic

### Machine Learning

- Scikit-learn
- Pandas
- NumPy
- Joblib

### Testing & Infrastructure

- `pytest` — API testing suite
- Vercel Serverless Functions

## Project Structure

```text
.
├── app/
│   ├── api/
│   │   └── routes.py           # FastAPI endpoints for health, metadata, and predictions
│   ├── schemas/
│   │   └── predict.py          # Pydantic models for request/response validation
│   ├── services/
│   │   └── model_loader.py     # Lazy-loading service for models and encoders
│   ├── config.py               # Application configuration and CORS settings
│   └── main.py                 # FastAPI application entry point
├── data/
│   ├── clean_cards.csv         # Processed training dataset
│   └── raw_cards.json          # Raw card dataset from YGOPRODeck API
├── models/
│   ├── archetype_model.joblib  # Serialized Random Forest classifier
│   ├── label_encoders.joblib   # Serialized feature encoders
│   └── target_encoder.joblib   # Serialized archetype target encoder
├── pipeline/
│   ├── clean_data.py           # Data cleaning and feature engineering
│   ├── fetch_data.py           # Script to fetch raw card data from API
│   └── train_model.py          # Model training and serialization script
├── tests/
│   └── test_api.py             # Integration and endpoint tests using pytest
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt            # Python dependencies
└── vercel.json                 # Vercel serverless deployment configuration
```

## Machine Learning Pipeline

The model is built through a three-stage workflow:

### 1. Data Collection

Download the latest monster card data from the YGOPRODeck API.

### 2. Data Preparation

Filter unsupported records, clean missing values, encode categorical features, and generate the training dataset.

### 3. Model Training

Train a Random Forest classifier and serialize the trained model together with its label encoders for production inference.

The pipeline scripts are intended to be executed manually whenever a new dataset or model version is required.

## API Endpoints

| Endpoint | Description |
|-----------|-------------|
| `/api/health` | Returns the service health status |
| `/api/metadata` | Returns valid card types, races, and attributes |
| `/api/predict` | Predicts the most likely archetype and confidence scores |

## Performance & Reliability

To improve serverless performance, trained models are loaded only when the first prediction request is received and then cached for subsequent requests.

The backend also validates incoming payloads using Pydantic, restricts cross-origin requests to the production frontend, and sanitizes numerical inputs before inference.

## Security

Only the production frontend is permitted through the CORS policy.

Incoming requests undergo validation before reaching the machine learning model, while internal exceptions are logged server-side without exposing implementation details to API consumers.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.
