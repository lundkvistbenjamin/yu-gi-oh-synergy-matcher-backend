from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import ALLOWED_ORIGINS
from app.api.routes import router

app = FastAPI(
    title="Duelist Synergy API",
    docs_url=None, # Disables automatic Swagger UI documentation for production security
    redoc_url=None # Disables ReDoc documentation
)

# 1. CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"], # Explicitly allow OPTIONS for CORS preflight
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(router)