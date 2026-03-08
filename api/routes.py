from fastapi import APIRouter
import pandas as pd
from api.schemas import StudentData, PredictionResponse
from api.services.model_services import ml_models

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict_risk(student: StudentData):
    try:
        model = ml_models.get("passos_magicos_model")
        scaler = ml_models.get("passos_magicos_scaler")
        
        if not model or not scaler:
            raise ValueError("Modelo ou Scaler não estão carregados na memória.")

        input_data = pd.DataFrame([student.model_dump()])
        
        input_scaled = scaler.transform(input_data)
        
        # 4. Faz a predição (Descobre o Cluster)
        prediction = model.predict(input_scaled)

        return PredictionResponse(
            classe_predita=int(prediction[0]), # Retornará 0, 1, 2 ou 3 (o Cluster)
            metodo="machine_learning_kmeans"
        )
    except Exception as e:
        return PredictionResponse(
            classe_predita=-1, # Indica erro
            metodo=f"Erro: {str(e)}"
        )