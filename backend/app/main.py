from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routes import health, predict
from app.config import settings
from app.models.model_loader import model_loader

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load AI models on startup
    model_loader.load_all_models()
    yield
    # Clean up (if necessary) on shutdown

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API for AI Fake Brand Detection System",
    version=settings.VERSION,
    lifespan=lifespan
)

# Standard CORS setup to allow Next.js frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to the Next.js origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(predict.router, prefix="/predict", tags=["predict"])
