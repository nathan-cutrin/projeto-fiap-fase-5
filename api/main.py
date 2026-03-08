from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routes import router
from api.services.model_services import load_ml_artifacts, ml_models

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_ml_artifacts()
    except Exception as e:
        print(f"Aviso no Startup: Artefatos de ML não carregados. Fallback ativado. Erro: {e}")
        pass
        
    yield 
    ml_models.clear()

app = FastAPI(
    title="API Passos Mágicos", 
    description="API para clusterização e identificação de perfis de alunos usando K-Means",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)