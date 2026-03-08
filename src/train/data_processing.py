import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import pandas as pd
from pathlib import Path

def carregar_dados():
    """
    Função para carregar os dados das abas do arquivo Excel e retorná-las como DataFrames.
    """
    # Obtém o diretório raiz do projeto
    current_directory = Path(__file__).resolve().parent.parent.parent  # Subindo 3 níveis para a raiz do projeto

    # Caminho para acessar o arquivo Excel em 'data/raw'
    file_path = current_directory / 'data' / 'raw' / 'base_dados_passos_magicos.xlsx'

    # Carregar os dados de todas as abas do Excel
    try:
        dicionario_abas = pd.read_excel(file_path, sheet_name=None)
        print(f"Arquivo carregado com sucesso! Caminho: {file_path}")
        
        # Carregar as abas específicas para 2022, 2023 e 2024
        df_2022 = dicionario_abas['PEDE2022'].copy()
        df_2023 = dicionario_abas['PEDE2023'].copy()
        df_2024 = dicionario_abas['PEDE2024'].copy()

        # Retornar os 3 DataFrames
        return df_2022, df_2023, df_2024
    except FileNotFoundError:
        print(f"Erro: O arquivo não foi encontrado em {file_path}")
        return None, None, None
    except KeyError as e:
        print(f"Erro: A aba {e} não foi encontrada no arquivo Excel.")
        return None, None, None

def renomear_colunas(df_2022, df_2023, df_2024):
    """
    Renomear as colunas para um formato padrão.
    """
    colunas_finais = {
        'IDA': 'indicador_desempenho_academico',
        'IEG': 'indicador_engajamento',
        'IPS': 'indicador_psicossocial',
        'IAA': 'indicador_autoavaliacao'
    }

    df_2022 = df_2022.rename(columns=colunas_finais)
    df_2023 = df_2023.rename(columns=colunas_finais)
    df_2024 = df_2024.rename(columns=colunas_finais)

    return df_2022, df_2023, df_2024

def preparar_dados(df_2022, df_2023, df_2024):
    """
    Filtra os dados e prepara as variáveis para treinamento.
    """
    features_kmeans = [
        'indicador_desempenho_academico',
        'indicador_engajamento',
        'indicador_psicossocial',
        'indicador_autoavaliacao'
    ]
    
    df_2022['origem_aba'] = 2022
    df_2023['origem_aba'] = 2023
    df_2024['origem_aba'] = 2024
    
    df_lista = [df_2022, df_2023, df_2024]
    
    for i in range(len(df_lista)):
        df_lista[i] = df_lista[i][features_kmeans + ['origem_aba']].dropna()

    df = pd.concat(df_lista, ignore_index=True)
    
    df = df[df['indicador_autoavaliacao'] != 0].reset_index(drop=True)
    
    # Filtrar dados de 2024
    df_2024 = df[df['origem_aba'] == 2024].copy()
    
    # Calcular as dimensões
    df_2024['dimensao_academica'] = df_2024[['indicador_desempenho_academico', 'indicador_engajamento']].mean(axis=1)
    df_2024['dimensao_psicossocial'] = df_2024[['indicador_psicossocial', 'indicador_autoavaliacao']].mean(axis=1)

    return df_2024

def padronizar_dados(df_2024):
    """
    Função para padronizar as variáveis de entrada.
    """
    features_set = ['dimensao_academica', 'dimensao_psicossocial']
    df_hierarquico = df_2024[features_set].dropna().copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_hierarquico)
    
    return X_scaled, scaler