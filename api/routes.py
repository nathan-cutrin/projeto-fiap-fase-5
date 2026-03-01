from fastapi import APIRouter
import pandas as pd
from api.schemas import StudentData, PredictionResponse
from api.services.model_services import ml_models

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict_risk(student: StudentData):
    try:
        model = ml_models.get("passos_magicos_model")
        
        if not model:
            raise ValueError()

        input_data = pd.DataFrame([student.model_dump()])
        
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        return PredictionResponse(
            risco_predito=int(prediction[0]),
            probabilidade=float(probability),
            metodo="machine_learning"
        )

    except Exception:
        risco_estimado = 1 if student.faltas > 15 else 0
        
        return PredictionResponse(
            risco_predito=risco_estimado,
            probabilidade=None,
            metodo="heuristica_fallback"
        )