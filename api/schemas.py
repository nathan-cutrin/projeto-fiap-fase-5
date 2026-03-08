from pydantic import BaseModel

class StudentData(BaseModel):
    dimensao_academica: float
    dimensao_psicossocial: float

class PredictionResponse(BaseModel):
    classe_predita: int
    metodo: str