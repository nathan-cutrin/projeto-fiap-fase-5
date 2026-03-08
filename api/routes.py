from fastapi import APIRouter, HTTPException
import pandas as pd
from api.schemas import StudentData, PredictionResponse, StudentResponse
from api.services.model_services import ml_models

router = APIRouter()

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