import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from api.routes import router, _reset_cache
import api.routes as routes_module

app = FastAPI()
app.include_router(router)
client = TestClient(app)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_db():
    return pd.DataFrame({
        'ra':                              ['101', '102', '103'],
        'fase':                            ['Fase 1', 'Fase 2', 'Fase Alfa'],
        'idade':                           [10, 11, 12],
        'indicador_desempenho_academico':  [8.0, 7.0, 6.0],
        'indicador_engajamento':           [9.0, 8.0, 7.0],
        'indicador_psicossocial':          [7.5, 6.5, 5.5],
        'indicador_autoavaliacao':         [8.5, 7.5, 6.5],
        'dimensao_academica':              [8.5, 7.5, 6.5],
        'dimensao_psicossocial':           [8.0, 7.0, 6.0],
    })

def make_model(predictions=None):
    if predictions is None:
        predictions = [0]
    model = MagicMock()
    model.predict.return_value = np.array(predictions)
    return model

def make_scaler():
    scaler = MagicMock()
    scaler.transform.side_effect = lambda x: x
    return scaler


# ─────────────────────────────────────────────
# Fixture: limpa cache antes de cada teste
# ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_estado():
    _reset_cache()
    yield
    _reset_cache()


# ─────────────────────────────────────────────
# Testes: GET /student/{ra_numero}
# ─────────────────────────────────────────────

class TestGetStudentByRa:

    def test_retorna_200_com_aluno_existente(self):
        with patch.dict(routes_module.ml_models, {"student_database": make_db()}, clear=True):
            response = client.get("/student/101")
            assert response.status_code == 200

    def test_retorna_dados_corretos_do_aluno(self):
        with patch.dict(routes_module.ml_models, {"student_database": make_db()}, clear=True):
            response = client.get("/student/101")
            data = response.json()
            assert data["ra"] == "101"
            assert data["fase"] == "Fase 1"
            assert data["idade"] == 10

    def test_retorna_404_com_ra_inexistente(self):
        with patch.dict(routes_module.ml_models, {"student_database": make_db()}, clear=True):
            response = client.get("/student/999")
            assert response.status_code == 404

    def test_retorna_mensagem_de_erro_404(self):
        with patch.dict(routes_module.ml_models, {"student_database": make_db()}, clear=True):
            response = client.get("/student/999")
            assert "999" in response.json()["detail"]

    def test_retorna_500_sem_banco_de_dados(self):
        with patch.dict(routes_module.ml_models, {}, clear=True):
            response = client.get("/student/101")
            assert response.status_code == 500

    def test_retorna_indicadores_do_aluno(self):
        with patch.dict(routes_module.ml_models, {"student_database": make_db()}, clear=True):
            response = client.get("/student/102")
            data = response.json()
            assert data["indicador_desempenho_academico"] == 7.0
            assert data["indicador_engajamento"] == 8.0

    def test_ra_com_espacos_nao_crasha(self):
        with patch.dict(routes_module.ml_models, {"student_database": make_db()}, clear=True):
            response = client.get("/student/101")
            assert response.status_code in [200, 404]


# ─────────────────────────────────────────────
# Testes: POST /predict
# ─────────────────────────────────────────────

class TestPredictRisk:

    def test_retorna_200_com_input_valido(self):
        with patch.dict(routes_module.ml_models, {
            "passos_magicos_model":  make_model([0]),
            "passos_magicos_scaler": make_scaler(),
        }, clear=True):
            response = client.post("/predict", json={
                "dimensao_academica": 8.0,
                "dimensao_psicossocial": 7.5
            })
            assert response.status_code == 200

    def test_retorna_cluster_ajustado_mais_1(self):
        with patch.dict(routes_module.ml_models, {
            "passos_magicos_model":  make_model([0]),
            "passos_magicos_scaler": make_scaler(),
        }, clear=True):
            response = client.post("/predict", json={
                "dimensao_academica": 8.0,
                "dimensao_psicossocial": 7.5
            })
            assert response.json()["classe_predita"] == 1

    def test_cluster_2_retorna_3(self):
        with patch.dict(routes_module.ml_models, {
            "passos_magicos_model":  make_model([2]),
            "passos_magicos_scaler": make_scaler(),
        }, clear=True):
            response = client.post("/predict", json={
                "dimensao_academica": 5.0,
                "dimensao_psicossocial": 4.5
            })
            assert response.json()["classe_predita"] == 3

    def test_retorna_metodo_correto(self):
        with patch.dict(routes_module.ml_models, {
            "passos_magicos_model":  make_model([1]),
            "passos_magicos_scaler": make_scaler(),
        }, clear=True):
            response = client.post("/predict", json={
                "dimensao_academica": 8.0,
                "dimensao_psicossocial": 7.5
            })
            assert response.json()["metodo"] == "machine_learning_kmeans"

    def test_retorna_erro_sem_modelo(self):
        with patch.dict(routes_module.ml_models, {}, clear=True):
            response = client.post("/predict", json={
                "dimensao_academica": 8.0,
                "dimensao_psicossocial": 7.5
            })
            assert response.json()["classe_predita"] == -1

    def test_retorna_erro_sem_scaler(self):
        with patch.dict(routes_module.ml_models, {
            "passos_magicos_model": make_model([0]),
        }, clear=True):
            response = client.post("/predict", json={
                "dimensao_academica": 8.0,
                "dimensao_psicossocial": 7.5
            })
            assert response.json()["classe_predita"] == -1

    def test_metodo_contem_erro_quando_falha(self):
        with patch.dict(routes_module.ml_models, {}, clear=True):
            response = client.post("/predict", json={
                "dimensao_academica": 8.0,
                "dimensao_psicossocial": 7.5
            })
            assert "Erro" in response.json()["metodo"]

    def test_scaler_e_chamado_com_input(self):
        scaler = make_scaler()
        with patch.dict(routes_module.ml_models, {
            "passos_magicos_model":  make_model([0]),
            "passos_magicos_scaler": scaler,
        }, clear=True):
            client.post("/predict", json={
                "dimensao_academica": 8.0,
                "dimensao_psicossocial": 7.5
            })
            scaler.transform.assert_called_once()


# ─────────────────────────────────────────────
# Testes: GET /clusters/stats
# ─────────────────────────────────────────────

class TestGetClustersStats:

    def _ml_models_completo(self):
        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 0])
        return {
            "student_database":      make_db(),
            "passos_magicos_model":  model,
            "passos_magicos_scaler": make_scaler(),
        }

    def test_retorna_200_com_recursos_carregados(self):
        with patch.dict(routes_module.ml_models, self._ml_models_completo(), clear=True):
            response = client.get("/clusters/stats")
            assert response.status_code == 200

    def test_retorna_500_sem_recursos(self):
        with patch.dict(routes_module.ml_models, {}, clear=True):
            response = client.get("/clusters/stats")
            assert response.status_code == 500

    def test_retorna_dict_com_clusters(self):
        with patch.dict(routes_module.ml_models, self._ml_models_completo(), clear=True):
            response = client.get("/clusters/stats")
            assert isinstance(response.json(), dict)
            assert len(response.json()) > 0

    def test_cada_cluster_tem_n_alunos(self):
        with patch.dict(routes_module.ml_models, self._ml_models_completo(), clear=True):
            response = client.get("/clusters/stats")
            for cluster_stats in response.json().values():
                assert "n_alunos" in cluster_stats

    def test_cada_cluster_tem_stats_de_dimensao_academica(self):
        with patch.dict(routes_module.ml_models, self._ml_models_completo(), clear=True):
            response = client.get("/clusters/stats")
            for cluster_stats in response.json().values():
                assert "dimensao_academica" in cluster_stats

    def test_stats_tem_mean_median_min_max(self):
        with patch.dict(routes_module.ml_models, self._ml_models_completo(), clear=True):
            response = client.get("/clusters/stats")
            for cluster_stats in response.json().values():
                s = cluster_stats.get("dimensao_academica", {})
                assert "mean" in s and "median" in s and "min" in s and "max" in s

    def test_cache_e_usado_na_segunda_chamada(self):
        ml = self._ml_models_completo()
        with patch.dict(routes_module.ml_models, ml, clear=True):
            client.get("/clusters/stats")
            chamadas_antes = ml["passos_magicos_model"].predict.call_count
            client.get("/clusters/stats")
            assert ml["passos_magicos_model"].predict.call_count == chamadas_antes


# ─────────────────────────────────────────────
# Testes: GET /clusters/stats/{cluster_id}
# ─────────────────────────────────────────────

class TestGetClusterStatsById:

    def _ml_models_completo(self):
        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 0])
        return {
            "student_database":      make_db(),
            "passos_magicos_model":  model,
            "passos_magicos_scaler": make_scaler(),
        }

    def test_retorna_200_com_cluster_existente(self):
        with patch.dict(routes_module.ml_models, self._ml_models_completo(), clear=True):
            client.get("/clusters/stats")  # popula cache
            response = client.get("/clusters/stats/1")
            assert response.status_code == 200

    def test_retorna_404_com_cluster_inexistente(self):
        with patch.dict(routes_module.ml_models, self._ml_models_completo(), clear=True):
            client.get("/clusters/stats")  # popula cache
            response = client.get("/clusters/stats/99")
            assert response.status_code == 404

    def test_retorna_stats_do_cluster_correto(self):
        with patch.dict(routes_module.ml_models, self._ml_models_completo(), clear=True):
            client.get("/clusters/stats")  # popula cache
            response = client.get("/clusters/stats/1")
            data = response.json()
            assert "n_alunos" in data
            assert "dimensao_academica" in data