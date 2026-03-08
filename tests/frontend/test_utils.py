import pytest
import sys
import os

# Permite importar o utils sem precisar instalar o pacote
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'frontend'))

from utils import converter_stats, montar_rows_stats


# ─────────────────────────────────────────────
# Fixtures reutilizáveis
# ─────────────────────────────────────────────

@pytest.fixture
def stats_completo():
    """Stats com todas as 6 colunas esperadas."""
    return {
        "indicador_desempenho_academico": {"mean": 7.0, "median": 7.5, "min": 3.0, "max": 10.0},
        "indicador_engajamento":          {"mean": 8.0, "median": 8.5, "min": 4.0, "max": 10.0},
        "indicador_psicossocial":         {"mean": 6.5, "median": 6.0, "min": 2.5, "max": 9.0},
        "indicador_autoavaliacao":        {"mean": 7.2, "median": 7.0, "min": 3.0, "max": 10.0},
        "dimensao_academica":             {"mean": 7.5, "median": 7.8, "min": 3.5, "max": 10.0},
        "dimensao_psicossocial":          {"mean": 6.8, "median": 6.5, "min": 2.5, "max": 9.5},
    }

@pytest.fixture
def stats_parcial():
    """Stats com apenas 2 colunas."""
    return {
        "indicador_desempenho_academico": {"mean": 7.0, "median": 7.5, "min": 3.0, "max": 10.0},
        "dimensao_academica":             {"mean": 7.5, "median": 7.8, "min": 3.5, "max": 10.0},
    }


# ─────────────────────────────────────────────
# Testes: converter_stats
# ─────────────────────────────────────────────

class TestConverterStats:

    def test_converte_chaves_string_para_int(self):
        entrada = {"1": {"n_alunos": 10}, "2": {"n_alunos": 20}}
        resultado = converter_stats(entrada)
        assert 1 in resultado
        assert 2 in resultado

    def test_nao_contem_chaves_string_apos_conversao(self):
        entrada = {"1": {"n_alunos": 10}}
        resultado = converter_stats(entrada)
        assert "1" not in resultado

    def test_valores_preservados_apos_conversao(self):
        entrada = {"3": {"n_alunos": 42, "dimensao_academica": {"mean": 8.0}}}
        resultado = converter_stats(entrada)
        assert resultado[3]["n_alunos"] == 42

    def test_entrada_vazia_retorna_dict_vazio(self):
        assert converter_stats({}) == {}

    def test_multiplos_clusters(self):
        entrada = {"1": {}, "2": {}, "3": {}}
        resultado = converter_stats(entrada)
        assert set(resultado.keys()) == {1, 2, 3}

    def test_chave_zero(self):
        entrada = {"0": {"n_alunos": 5}}
        resultado = converter_stats(entrada)
        assert 0 in resultado


# ─────────────────────────────────────────────
# Testes: montar_rows_stats
# ─────────────────────────────────────────────

class TestMontarRowsStats:

    def test_retorna_lista(self, stats_completo):
        resultado = montar_rows_stats(stats_completo)
        assert isinstance(resultado, list)

    def test_retorna_6_rows_com_stats_completo(self, stats_completo):
        resultado = montar_rows_stats(stats_completo)
        assert len(resultado) == 6

    def test_retorna_2_rows_com_stats_parcial(self, stats_parcial):
        resultado = montar_rows_stats(stats_parcial)
        assert len(resultado) == 2

    def test_chaves_corretas_em_cada_row(self, stats_completo):
        rows = montar_rows_stats(stats_completo)
        for row in rows:
            assert "Indicador" in row
            assert "Média"     in row
            assert "Mediana"   in row
            assert "Mín"       in row
            assert "Máx"       in row

    def test_labels_corretos(self, stats_completo):
        rows = montar_rows_stats(stats_completo)
        labels = [r["Indicador"] for r in rows]
        assert "Desemp. Acadêmico" in labels
        assert "Engajamento"       in labels
        assert "Psicossocial"      in labels
        assert "Autoavaliação"     in labels
        assert "Dim. Acadêmica"    in labels
        assert "Dim. Psicossocial" in labels

    def test_valores_numericos_corretos(self, stats_completo):
        rows = montar_rows_stats(stats_completo)
        row_desempenho = next(r for r in rows if r["Indicador"] == "Desemp. Acadêmico")
        assert row_desempenho["Média"]   == 7.0
        assert row_desempenho["Mediana"] == 7.5
        assert row_desempenho["Mín"]     == 3.0
        assert row_desempenho["Máx"]     == 10.0

    def test_ignora_chaves_desconhecidas(self):
        stats = {"chave_que_nao_existe": {"mean": 1.0, "median": 1.0, "min": 0.0, "max": 2.0}}
        resultado = montar_rows_stats(stats)
        assert resultado == []

    def test_entrada_vazia_retorna_lista_vazia(self):
        assert montar_rows_stats({}) == []

    def test_apenas_colunas_conhecidas_sao_incluidas(self):
        stats = {
            "indicador_engajamento": {"mean": 8.0, "median": 8.0, "min": 5.0, "max": 10.0},
            "coluna_desconhecida":   {"mean": 1.0, "median": 1.0, "min": 0.0, "max": 2.0},
        }
        rows = montar_rows_stats(stats)
        assert len(rows) == 1
        assert rows[0]["Indicador"] == "Engajamento"