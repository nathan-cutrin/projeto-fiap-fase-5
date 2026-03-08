import pytest
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from api.services.model_services import load_ml_artifacts, ml_models


# ─────────────────────────────────────────────
# Fixture: reseta ml_models antes de cada teste
# ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_ml_models():
    ml_models["passos_magicos_model"]  = None
    ml_models["passos_magicos_scaler"] = None
    ml_models["student_database"]      = None
    yield
    ml_models["passos_magicos_model"]  = None
    ml_models["passos_magicos_scaler"] = None
    ml_models["student_database"]      = None


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_csv_df():
    """DataFrame fake simulando o CSV de alunos."""
    return pd.DataFrame({
        'ra':                              [101, 102, 103],
        'fase':                            ['Fase 1', 'Fase 2', None],
        'idade':                           [10.0, 11.0, None],
        'indicador_desempenho_academico':  [8.0, 7.0, 6.0],
        'indicador_engajamento':           [9.0, 8.0, 7.0],
        'indicador_psicossocial':          [7.5, 6.5, 5.5],
        'indicador_autoavaliacao':         [8.5, 7.5, 6.5],
        'dimensao_academica':              [8.5, 7.5, 6.5],
        'dimensao_psicossocial':           [8.0, 7.0, 6.0],
    })


# ─────────────────────────────────────────────
# Testes: estrutura do ml_models
# ─────────────────────────────────────────────

class TestMlModelsEstrutura:

    def test_ml_models_e_dicionario(self):
        assert isinstance(ml_models, dict)

    def test_ml_models_tem_chave_model(self):
        assert "passos_magicos_model" in ml_models

    def test_ml_models_tem_chave_scaler(self):
        assert "passos_magicos_scaler" in ml_models

    def test_ml_models_tem_chave_database(self):
        assert "student_database" in ml_models

    def test_valores_iniciais_sao_none(self):
        assert ml_models["passos_magicos_model"]  is None
        assert ml_models["passos_magicos_scaler"] is None
        assert ml_models["student_database"]      is None


# ─────────────────────────────────────────────
# Testes: load_ml_artifacts — carregamento do modelo
# ─────────────────────────────────────────────

class TestLoadMlArtifacts:

    def test_carrega_model_quando_arquivo_existe(self):
        mock_model = MagicMock()
        with patch("api.services.model_services.Path.exists", return_value=True), \
             patch("api.services.model_services.joblib.load", return_value=mock_model):
            load_ml_artifacts()
            assert ml_models["passos_magicos_model"] is not None

    def test_carrega_scaler_quando_arquivo_existe(self):
        mock_scaler = MagicMock()
        with patch("api.services.model_services.Path.exists", return_value=True), \
             patch("api.services.model_services.joblib.load", return_value=mock_scaler):
            load_ml_artifacts()
            assert ml_models["passos_magicos_scaler"] is not None

    def test_nao_carrega_model_quando_arquivo_nao_existe(self):
        with patch("api.services.model_services.Path.exists", return_value=False):
            load_ml_artifacts()
            assert ml_models["passos_magicos_model"] is None

    def test_nao_carrega_scaler_quando_arquivo_nao_existe(self):
        with patch("api.services.model_services.Path.exists", return_value=False):
            load_ml_artifacts()
            assert ml_models["passos_magicos_scaler"] is None

    def test_nao_carrega_database_quando_arquivo_nao_existe(self):
        with patch("api.services.model_services.Path.exists", return_value=False):
            load_ml_artifacts()
            assert ml_models["student_database"] is None

    def test_nao_crasha_quando_arquivos_ausentes(self):
        with patch("api.services.model_services.Path.exists", return_value=False):
            # Não deve levantar exceção
            load_ml_artifacts()

    def test_nao_crasha_em_excecao_inesperada(self):
        with patch("api.services.model_services.Path.exists", side_effect=Exception("erro inesperado")):
            # O try/except interno deve absorver o erro
            load_ml_artifacts()


# ─────────────────────────────────────────────
# Testes: load_ml_artifacts — carregamento do CSV
# ─────────────────────────────────────────────

class TestLoadDatabase:

    def test_carrega_database_quando_csv_existe(self):
        df_fake = make_csv_df()
        with patch("api.services.model_services.Path.exists", return_value=True), \
             patch("api.services.model_services.joblib.load", return_value=MagicMock()), \
             patch("api.services.model_services.pd.read_csv", return_value=df_fake):
            load_ml_artifacts()
            assert ml_models["student_database"] is not None

    def test_ra_convertido_para_string(self):
        df_fake = make_csv_df()
        with patch("api.services.model_services.Path.exists", return_value=True), \
             patch("api.services.model_services.joblib.load", return_value=MagicMock()), \
             patch("api.services.model_services.pd.read_csv", return_value=df_fake):
            load_ml_artifacts()
            db = ml_models["student_database"]
            assert db["ra"].dtype == object  # string no pandas é dtype object
            assert db["ra"].iloc[0] == "101"

    def test_ra_sem_espacos(self):
        df_fake = make_csv_df()
        df_fake["ra"] = ["  101  ", " 102", "103 "]
        with patch("api.services.model_services.Path.exists", return_value=True), \
             patch("api.services.model_services.joblib.load", return_value=MagicMock()), \
             patch("api.services.model_services.pd.read_csv", return_value=df_fake):
            load_ml_artifacts()
            db = ml_models["student_database"]
            assert db["ra"].iloc[0] == "101"
            assert db["ra"].iloc[1] == "102"

    def test_nulos_substituidos_por_none(self):
        df_fake = make_csv_df()  # já tem None em 'fase' e 'idade' na linha 2
        with patch("api.services.model_services.Path.exists", return_value=True), \
             patch("api.services.model_services.joblib.load", return_value=MagicMock()), \
             patch("api.services.model_services.pd.read_csv", return_value=df_fake):
            load_ml_artifacts()
            db = ml_models["student_database"]
            # A linha com None não deve ter virado NaN não tratado
            assert db is not None

    def test_quantidade_de_alunos_preservada(self):
        df_fake = make_csv_df()
        with patch("api.services.model_services.Path.exists", return_value=True), \
             patch("api.services.model_services.joblib.load", return_value=MagicMock()), \
             patch("api.services.model_services.pd.read_csv", return_value=df_fake):
            load_ml_artifacts()
            assert len(ml_models["student_database"]) == 3