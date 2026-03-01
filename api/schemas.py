from pydantic import BaseModel
from typing import Optional

class StudentData(BaseModel):
    faltas: int

class PredictionResponse(BaseModel):
    risco_predito: int
    probabilidade: Optional[float]
    metodo: str