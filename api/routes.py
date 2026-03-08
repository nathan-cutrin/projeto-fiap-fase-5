from fastapi import APIRouter, HTTPException
import pandas as pd
from api.schemas import StudentData, PredictionResponse, StudentResponse
from api.services.model_services import ml_models

router = APIRouter()
_stats_cache = None

@router.get("/student/{ra_numero}", response_model=StudentResponse)
def get_student_by_ra(ra_numero: str):
    db = ml_models.get("student_database")
    if db is None:
        raise HTTPException(status_code=500, detail="Banco de dados não carregado.")

    ra_busca = ra_numero.strip()
    filtro = db[db['ra'] == ra_busca]
    
    if filtro.empty:
        raise HTTPException(status_code=404, detail=f"Aluno com RA {ra_busca} não encontrado.")
    
    return StudentResponse(**filtro.iloc[0].to_dict())

@router.post("/predict", response_model=PredictionResponse)
def predict_risk(student: StudentData):
    try:
        model = ml_models.get("passos_magicos_model")
        scaler = ml_models.get("passos_magicos_scaler")
        
        if not model or not scaler:
            raise ValueError("Modelo ou Scaler não estão carregados na memória.")

        input_data = pd.DataFrame([student.model_dump()])
        
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled)

        return PredictionResponse(
            classe_predita=int(prediction[0]) + 1, # Ajuste para que as classes sejam 1, 2, 3 em vez de 0, 1, 2
            metodo="machine_learning_kmeans"
        )
    except Exception as e:
        return PredictionResponse(
            classe_predita=-1,
            metodo=f"Erro: {str(e)}"
        )
    
@router.get("/clusters/stats")
def get_clusters_stats():
    global _stats_cache
    if _stats_cache is not None:
        return _stats_cache

    db = ml_models.get("student_database")
    model = ml_models.get("passos_magicos_model")
    scaler = ml_models.get("passos_magicos_scaler")

    if db is None or model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Modelo ou base de dados não carregados.")

    df = db.copy()
    df["dimensao_academica"]    = df[["indicador_desempenho_academico", "indicador_engajamento"]].mean(axis=1)
    df["dimensao_psicossocial"] = df[["indicador_psicossocial", "indicador_autoavaliacao"]].mean(axis=1)

    features = df[["dimensao_academica", "dimensao_psicossocial"]].dropna()
    df.loc[features.index, "cluster"] = model.predict(scaler.transform(features)).astype(int) + 1  # mesmo ajuste do /predict
    df = df.dropna(subset=["cluster"])
    df["cluster"] = df["cluster"].astype(int)

    colunas_stats = [
        "indicador_desempenho_academico",
        "indicador_engajamento",
        "indicador_psicossocial",
        "indicador_autoavaliacao",
        "dimensao_academica",
        "dimensao_psicossocial",
    ]

    resultado = {}
    for cluster_id, grupo in df.groupby("cluster"):
        resultado[int(cluster_id)] = {
            col: {
                "mean":   round(grupo[col].mean(), 2),
                "median": round(grupo[col].median(), 2),
                "min":    round(grupo[col].min(), 2),
                "max":    round(grupo[col].max(), 2),
            }
            for col in colunas_stats if col in grupo.columns
        }
        resultado[int(cluster_id)]["n_alunos"] = len(grupo)

    _stats_cache = resultado
    return resultado


@router.get("/clusters/stats/{cluster_id}")
def get_cluster_stats(cluster_id: int):
    stats = get_clusters_stats()  
    if cluster_id not in stats:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} não encontrado.")
    return stats[cluster_id]