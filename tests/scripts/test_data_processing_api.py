import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

# Permite importar o módulo sem instalar o pacote
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

from data_processing_api import (
    padronizar_fase,
    renomear_colunas,
    preparar_dados_api,
    formatar_colunas_lower,
    exportar_base_api,
    carregar_dados,
)


# ─────────────────────────────────────────────
# Fixtures reutilizáveis
# ─────────────────────────────────────────────

@pytest.fixture
def df_base():
    """DataFrame mínimo com as 4 features e colunas demográficas."""
    return pd.DataFrame({
        'RA':                              ['101', '102', '103'],
        'Fase':                            ['Fase 1', 'Fase 2', 'Fase Alfa'],
        'Idade':                           [10, 11, 12],
        'indicador_desempenho_academico':  [8.0, 7.0, 6.0],
        'indicador_engajamento':           [9.0, 8.0, 7.0],
        'indicador_psicossocial':          [7.5, 6.5, 5.5],
        'indicador_autoavaliacao':         [8.5, 7.5, 6.5],
    })

@pytest.fixture
def df_com_nulos():
    """DataFrame com alguns valores nulos nas features."""
    return pd.DataFrame({
        'RA':                              ['101', '102', '103'],
        'Fase':                            ['Fase 1', 'Fase 2', 'Fase 3'],
        'Idade':                           [10, 11, 12],
        'indicador_desempenho_academico':  [8.0,  None, 6.0],
        'indicador_engajamento':           [9.0,  8.0,  7.0],
        'indicador_psicossocial':          [7.5,  6.5,  5.5],
        'indicador_autoavaliacao':         [8.5,  7.5,  6.5],
    })

@pytest.fixture
def df_com_autoavaliacao_zero():
    """DataFrame com autoavaliação = 0 (deve ser removido)."""
    return pd.DataFrame({
        'RA':                              ['101', '102'],
        'Fase':                            ['Fase 1', 'Fase 2'],
        'Idade':                           [10, 11],
        'indicador_desempenho_academico':  [8.0, 7.0],
        'indicador_engajamento':           [9.0, 8.0],
        'indicador_psicossocial':          [7.5, 6.5],
        'indicador_autoavaliacao':         [0.0, 7.5],  # primeiro deve ser removido
    })

@pytest.fixture
def df_para_renomear():
    """DataFrame com colunas originais do Excel (IDA, IEG, IPS, IAA)."""
    return pd.DataFrame({
        'RA':    ['101'],
        'IDA':   [8.0],
        'IEG':   [9.0],
        'IPS':   [7.5],
        'IAA':   [8.5],
        'Outras': [1.0],
    })

@pytest.fixture
def tres_dfs(df_base):
    """Três DataFrames iguais simulando 2022, 2023, 2024."""
    df1 = df_base.copy()
    df2 = df_base.copy()
    df3 = df_base.copy()
    return df1, df2, df3


# ─────────────────────────────────────────────
# Testes: padronizar_fase
# ─────────────────────────────────────────────

class TestPadronizarFase:

    def test_fase_alfa_string(self):
        assert padronizar_fase('Fase Alfa') == 'Fase Alfa'

    def test_alfa_minusculo(self):
        assert padronizar_fase('alfa') == 'Fase Alfa'

    def test_fase_0_vira_alfa(self):
        assert padronizar_fase('0') == 'Fase Alfa'

    def test_fase_0_texto(self):
        assert padronizar_fase('FASE 0') == 'Fase Alfa'

    def test_fase_numerica_simples(self):
        assert padronizar_fase('Fase 1') == 'Fase 1'

    def test_fase_numero_sem_texto(self):
        assert padronizar_fase('2') == 'Fase 2'

    def test_fase_com_letra(self):
        assert padronizar_fase('1A') == 'Fase 1'

    def test_fase_maiuscula(self):
        assert padronizar_fase('FASE 3') == 'Fase 3'

    def test_fase_com_espacos(self):
        assert padronizar_fase('  Fase 4  ') == 'Fase 4'

    def test_fase_7(self):
        assert padronizar_fase('Fase 7') == 'Fase 7'

    def test_fase_alfa_com_espaco(self):
        assert padronizar_fase('  ALFA  ') == 'Fase Alfa'

    def test_valor_sem_numero_nem_alfa_retorna_original(self):
        resultado = padronizar_fase('sem_numero')
        # Deve retornar o valor original (sem crash)
        assert resultado is not None


# ─────────────────────────────────────────────
# Testes: renomear_colunas
# ─────────────────────────────────────────────

class TestRenomearColunas:

    def test_colunas_renomeadas_corretamente(self, df_para_renomear):
        df1 = df_para_renomear.copy()
        df2 = df_para_renomear.copy()
        df3 = df_para_renomear.copy()

        r1, r2, r3 = renomear_colunas(df1, df2, df3)

        assert 'indicador_desempenho_academico' in r1.columns
        assert 'indicador_engajamento'          in r1.columns
        assert 'indicador_psicossocial'         in r1.columns
        assert 'indicador_autoavaliacao'        in r1.columns

    def test_colunas_originais_removidas(self, df_para_renomear):
        df1 = df_para_renomear.copy()
        df2 = df_para_renomear.copy()
        df3 = df_para_renomear.copy()

        r1, _, _ = renomear_colunas(df1, df2, df3)

        assert 'IDA' not in r1.columns
        assert 'IEG' not in r1.columns
        assert 'IPS' not in r1.columns
        assert 'IAA' not in r1.columns

    def test_colunas_nao_mapeadas_preservadas(self, df_para_renomear):
        df1 = df_para_renomear.copy()
        df2 = df_para_renomear.copy()
        df3 = df_para_renomear.copy()

        r1, _, _ = renomear_colunas(df1, df2, df3)

        assert 'Outras' in r1.columns
        assert 'RA'     in r1.columns

    def test_renomeia_todos_os_tres_dfs(self, df_para_renomear):
        df1 = df_para_renomear.copy()
        df2 = df_para_renomear.copy()
        df3 = df_para_renomear.copy()

        r1, r2, r3 = renomear_colunas(df1, df2, df3)

        for df in [r1, r2, r3]:
            assert 'indicador_desempenho_academico' in df.columns

    def test_valores_preservados_apos_renomear(self, df_para_renomear):
        df1 = df_para_renomear.copy()
        df2 = df_para_renomear.copy()
        df3 = df_para_renomear.copy()

        r1, _, _ = renomear_colunas(df1, df2, df3)

        assert r1['indicador_desempenho_academico'].iloc[0] == 8.0


# ─────────────────────────────────────────────
# Testes: preparar_dados_api
# ─────────────────────────────────────────────

class TestPrepararDadosApi:

    def test_retorna_apenas_dados_2024(self, df_base):
        df_2022 = df_base.copy()
        df_2023 = df_base.copy()
        df_2024 = df_base.copy()

        resultado = preparar_dados_api(df_2022, df_2023, df_2024)

        assert 'origem_aba' in resultado.columns
        assert (resultado['origem_aba'] == 2024).all()

    def test_remove_linhas_com_nulos_nas_features(self, df_com_nulos):
        df_2022 = pd.DataFrame(columns=df_com_nulos.columns)
        df_2023 = pd.DataFrame(columns=df_com_nulos.columns)
        df_2024 = df_com_nulos.copy()

        resultado = preparar_dados_api(df_2022, df_2023, df_2024)

        assert resultado['indicador_desempenho_academico'].isna().sum() == 0

    def test_remove_autoavaliacao_zero(self, df_com_autoavaliacao_zero):
        df_2022 = pd.DataFrame(columns=df_com_autoavaliacao_zero.columns)
        df_2023 = pd.DataFrame(columns=df_com_autoavaliacao_zero.columns)
        df_2024 = df_com_autoavaliacao_zero.copy()

        resultado = preparar_dados_api(df_2022, df_2023, df_2024)

        assert (resultado['indicador_autoavaliacao'] != 0).all()

    def test_calcula_dimensao_academica(self, df_base):
        df_2022 = pd.DataFrame(columns=df_base.columns)
        df_2023 = pd.DataFrame(columns=df_base.columns)
        df_2024 = df_base.copy()

        resultado = preparar_dados_api(df_2022, df_2023, df_2024)

        esperado = (df_base['indicador_desempenho_academico'].iloc[0] +
                    df_base['indicador_engajamento'].iloc[0]) / 2
        assert resultado['dimensao_academica'].iloc[0] == pytest.approx(esperado)

    def test_calcula_dimensao_psicossocial(self, df_base):
        df_2022 = pd.DataFrame(columns=df_base.columns)
        df_2023 = pd.DataFrame(columns=df_base.columns)
        df_2024 = df_base.copy()

        resultado = preparar_dados_api(df_2022, df_2023, df_2024)

        esperado = (df_base['indicador_psicossocial'].iloc[0] +
                    df_base['indicador_autoavaliacao'].iloc[0]) / 2
        assert resultado['dimensao_psicossocial'].iloc[0] == pytest.approx(esperado)

    def test_extrai_apenas_numeros_do_ra(self, df_base):
        df_2024 = df_base.copy()
        df_2024['RA'] = ['RA101', 'RA102', 'RA103']
        df_2022 = pd.DataFrame(columns=df_2024.columns)
        df_2023 = pd.DataFrame(columns=df_2024.columns)

        resultado = preparar_dados_api(df_2022, df_2023, df_2024)

        assert resultado['RA'].iloc[0] == '101'

    def test_colunas_dimensao_criadas(self, df_base):
        df_2022 = pd.DataFrame(columns=df_base.columns)
        df_2023 = pd.DataFrame(columns=df_base.columns)

        resultado = preparar_dados_api(df_2022, df_2023, df_base.copy())

        assert 'dimensao_academica'    in resultado.columns
        assert 'dimensao_psicossocial' in resultado.columns


# ─────────────────────────────────────────────
# Testes: formatar_colunas_lower
# ─────────────────────────────────────────────

class TestFormatarColunasLower:

    def test_converte_para_minusculo(self):
        df = pd.DataFrame(columns=['NOME', 'IDADE', 'RA'])
        resultado = formatar_colunas_lower(df)
        assert list(resultado.columns) == ['nome', 'idade', 'ra']

    def test_remove_espacos_nas_bordas(self):
        df = pd.DataFrame(columns=[' nome ', ' idade '])
        resultado = formatar_colunas_lower(df)
        assert 'nome'  in resultado.columns
        assert 'idade' in resultado.columns

    def test_substitui_espacos_internos_por_underscore(self):
        df = pd.DataFrame(columns=['Nome Aluno', 'Data Nascimento'])
        resultado = formatar_colunas_lower(df)
        assert 'nome_aluno'        in resultado.columns
        assert 'data_nascimento'   in resultado.columns

    def test_colunas_ja_minusculas_nao_mudam(self):
        df = pd.DataFrame(columns=['nome', 'idade'])
        resultado = formatar_colunas_lower(df)
        assert list(resultado.columns) == ['nome', 'idade']

    def test_retorna_dataframe(self):
        df = pd.DataFrame(columns=['NOME'])
        resultado = formatar_colunas_lower(df)
        assert isinstance(resultado, pd.DataFrame)

    def test_valores_preservados(self):
        df = pd.DataFrame({'NOME': ['João'], 'IDADE': [10]})
        resultado = formatar_colunas_lower(df)
        assert resultado['nome'].iloc[0] == 'João'
        assert resultado['idade'].iloc[0] == 10


# ─────────────────────────────────────────────
# Testes: carregar_dados (com mock de arquivo)
# ─────────────────────────────────────────────

class TestCarregarDados:

    def test_retorna_none_quando_arquivo_nao_encontrado(self):
        with patch('data_processing_api.pd.read_excel', side_effect=FileNotFoundError):
            df1, df2, df3 = carregar_dados()
            assert df1 is None
            assert df2 is None
            assert df3 is None

    def test_retorna_none_quando_aba_nao_encontrada(self):
        with patch('data_processing_api.pd.read_excel', side_effect=KeyError('PEDE2022')):
            df1, df2, df3 = carregar_dados()
            assert df1 is None
            assert df2 is None
            assert df3 is None

    def test_retorna_tres_dataframes_quando_sucesso(self):
        mock_abas = {
            'PEDE2022': pd.DataFrame({'col': [1]}),
            'PEDE2023': pd.DataFrame({'col': [2]}),
            'PEDE2024': pd.DataFrame({'col': [3]}),
        }
        with patch('data_processing_api.pd.read_excel', return_value=mock_abas):
            df1, df2, df3 = carregar_dados()
            assert df1 is not None
            assert df2 is not None
            assert df3 is not None

    def test_dataframes_tem_conteudo_correto(self):
        mock_abas = {
            'PEDE2022': pd.DataFrame({'col': [2022]}),
            'PEDE2023': pd.DataFrame({'col': [2023]}),
            'PEDE2024': pd.DataFrame({'col': [2024]}),
        }
        with patch('data_processing_api.pd.read_excel', return_value=mock_abas):
            df1, df2, df3 = carregar_dados()
            assert df1['col'].iloc[0] == 2022
            assert df2['col'].iloc[0] == 2023
            assert df3['col'].iloc[0] == 2024


# ─────────────────────────────────────────────
# Testes: exportar_base_api (com mock de I/O)
# ─────────────────────────────────────────────

class TestExportarBaseApi:

    def test_chama_to_csv(self, df_base):
        df = df_base.copy()
        df.columns = df.columns.str.lower()
        df['dimensao_academica']    = 7.5
        df['dimensao_psicossocial'] = 7.0

        with patch('data_processing_api.Path.mkdir'), \
             patch('pandas.DataFrame.to_csv') as mock_csv:
            exportar_base_api(df)
            mock_csv.assert_called_once()

    def test_salva_apenas_colunas_existentes(self, df_base):
        """Não deve quebrar se uma coluna esperada não existir no df."""
        df = df_base.copy()
        df.columns = df.columns.str.lower()
        df['dimensao_academica']    = 7.5
        df['dimensao_psicossocial'] = 7.0
        # Remove a coluna 'fase' propositalmente
        df = df.drop(columns=['fase'], errors='ignore')

        with patch('data_processing_api.Path.mkdir'), \
             patch('pandas.DataFrame.to_csv'):
            # Não deve lançar exceção
            exportar_base_api(df)