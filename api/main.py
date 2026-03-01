from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routes import router
from api.services.model_services import load_ml_model, ml_models

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ml_models["passos_magicos_model"] = load_ml_model()
    except Exception:
        pass
    yield
    ml_models.clear()

app = FastAPI(title="API Passos Magicos", lifespan=lifespan)

app.include_router(router)