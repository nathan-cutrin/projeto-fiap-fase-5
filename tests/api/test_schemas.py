import pytest
import sys
import os
from pydantic import ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from api.schemas import StudentData, PredictionResponse, StudentResponse


# ─────────────────────────────────────────────
# Testes: StudentData
# ─────────────────────────────────────────────

class TestStudentData:

    def test_cria_com_campos_validos(self):
        student = StudentData(dimensao_academica=8.0, dimensao_psicossocial=7.5)
        assert student.dimensao_academica == 8.0
        assert student.dimensao_psicossocial == 7.5

    def test_aceita_valores_zero(self):
        student = StudentData(dimensao_academica=0.0, dimensao_psicossocial=0.0)
        assert student.dimensao_academica == 0.0
        assert student.dimensao_psicossocial == 0.0

    def test_aceita_valores_decimais(self):
        student = StudentData(dimensao_academica=7.333, dimensao_psicossocial=8.756)
        assert student.dimensao_academica == 7.333
        assert student.dimensao_psicossocial == 8.756

    def test_falha_sem_dimensao_academica(self):
        with pytest.raises(ValidationError):
            StudentData(dimensao_psicossocial=7.5)

    def test_falha_sem_dimensao_psicossocial(self):
        with pytest.raises(ValidationError):
            StudentData(dimensao_academica=8.0)

    def test_falha_com_campos_vazios(self):
        with pytest.raises(ValidationError):
            StudentData()

    def test_falha_com_string_no_lugar_de_float(self):
        with pytest.raises(ValidationError):
            StudentData(dimensao_academica="alto", dimensao_psicossocial=7.5)

    def test_model_dump_retorna_dict_correto(self):
        student = StudentData(dimensao_academica=8.0, dimensao_psicossocial=7.5)
        d = student.model_dump()
        assert d == {"dimensao_academica": 8.0, "dimensao_psicossocial": 7.5}


# ─────────────────────────────────────────────
# Testes: PredictionResponse
# ─────────────────────────────────────────────

class TestPredictionResponse:

    def test_cria_com_campos_validos(self):
        resp = PredictionResponse(classe_predita=1, metodo="machine_learning_kmeans")
        assert resp.classe_predita == 1
        assert resp.metodo == "machine_learning_kmeans"

    def test_aceita_classe_predita_negativa(self):
        resp = PredictionResponse(classe_predita=-1, metodo="Erro: modelo não carregado")
        assert resp.classe_predita == -1

    def test_aceita_qualquer_string_no_metodo(self):
        resp = PredictionResponse(classe_predita=2, metodo="Erro: falha inesperada")
        assert "Erro" in resp.metodo

    def test_falha_sem_classe_predita(self):
        with pytest.raises(ValidationError):
            PredictionResponse(metodo="kmeans")

    def test_falha_sem_metodo(self):
        with pytest.raises(ValidationError):
            PredictionResponse(classe_predita=1)

    def test_falha_com_string_em_classe_predita(self):
        with pytest.raises(ValidationError):
            PredictionResponse(classe_predita="um", metodo="kmeans")

    def test_model_dump_retorna_dict_correto(self):
        resp = PredictionResponse(classe_predita=3, metodo="machine_learning_kmeans")
        d = resp.model_dump()
        assert d["classe_predita"] == 3
        assert d["metodo"] == "machine_learning_kmeans"


# ─────────────────────────────────────────────
# Testes: StudentResponse
# ─────────────────────────────────────────────

class TestStudentResponse:

    def _dados_completos(self):
        return {
            "ra": "101",
            "fase": "Fase 1",
            "idade": 10.0,
            "indicador_desempenho_academico": 8.0,
            "indicador_engajamento": 9.0,
            "indicador_psicossocial": 7.5,
            "indicador_autoavaliacao": 8.5,
            "dimensao_academica": 8.5,
            "dimensao_psicossocial": 8.0,
        }

    def test_cria_com_todos_os_campos(self):
        resp = StudentResponse(**self._dados_completos())
        assert resp.ra == "101"
        assert resp.fase == "Fase 1"
        assert resp.dimensao_academica == 8.5

    def test_campos_opcionais_podem_ser_none(self):
        resp = StudentResponse(
            ra="101",
            dimensao_academica=8.5,
            dimensao_psicossocial=8.0,
        )
        assert resp.fase is None
        assert resp.idade is None
        assert resp.indicador_desempenho_academico is None
        assert resp.indicador_engajamento is None
        assert resp.indicador_psicossocial is None
        assert resp.indicador_autoavaliacao is None

    def test_falha_sem_ra(self):
        with pytest.raises(ValidationError):
            StudentResponse(dimensao_academica=8.5, dimensao_psicossocial=8.0)

    def test_falha_sem_dimensao_academica(self):
        with pytest.raises(ValidationError):
            StudentResponse(ra="101", dimensao_psicossocial=8.0)

    def test_falha_sem_dimensao_psicossocial(self):
        with pytest.raises(ValidationError):
            StudentResponse(ra="101", dimensao_academica=8.5)

    def test_aceita_indicadores_none_individualmente(self):
        resp = StudentResponse(
            ra="101",
            dimensao_academica=8.5,
            dimensao_psicossocial=8.0,
            indicador_desempenho_academico=None,
            indicador_engajamento=9.0,
        )
        assert resp.indicador_desempenho_academico is None
        assert resp.indicador_engajamento == 9.0

    def test_ra_e_string(self):
        resp = StudentResponse(ra="904", dimensao_academica=8.0, dimensao_psicossocial=7.0)
        assert isinstance(resp.ra, str)

    def test_model_dump_inclui_todos_os_campos(self):
        resp = StudentResponse(**self._dados_completos())
        d = resp.model_dump()
        assert "ra"                              in d
        assert "fase"                            in d
        assert "idade"                           in d
        assert "indicador_desempenho_academico"  in d
        assert "indicador_engajamento"           in d
        assert "indicador_psicossocial"          in d
        assert "indicador_autoavaliacao"         in d
        assert "dimensao_academica"              in d
        assert "dimensao_psicossocial"           in d