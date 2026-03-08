from pydantic import BaseModel
from typing import Optional
class StudentData(BaseModel):
    dimensao_academica: float
    dimensao_psicossocial: float

class PredictionResponse(BaseModel):
    classe_predita: int
    metodo: str
class StudentResponse(BaseModel):
    ra: str
    fase: Optional[str] = None
    idade: Optional[float] = None 
    indicador_desempenho_academico: Optional[float] = None
    indicador_engajamento: Optional[float] = None
    indicador_psicossocial: Optional[float] = None
    indicador_autoavaliacao: Optional[float] = None
    dimensao_academica: float
    dimensao_psicossocial: float