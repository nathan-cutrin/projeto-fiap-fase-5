import pandas as pd
from pathlib import Path
import re

def carregar_dados():
    """
    Função para carregar os dados das abas do arquivo Excel e retorná-las como DataFrames.
    """
    current_directory = Path(__file__).resolve().parent.parent

    file_path = current_directory / 'data' / 'raw' / 'base_dados_passos_magicos.xlsx'

    try:
        dicionario_abas = pd.read_excel(file_path, sheet_name=None)
        print(f"✅ Arquivo carregado com sucesso! Caminho: {file_path}")
        
        df_2022 = dicionario_abas['PEDE2022'].copy()
        df_2023 = dicionario_abas['PEDE2023'].copy()
        df_2024 = dicionario_abas['PEDE2024'].copy()

        return df_2022, df_2023, df_2024
        
    except FileNotFoundError:
        print(f"❌ Erro: O arquivo não foi encontrado em {file_path}")
        return None, None, None
    except KeyError as e:
        print(f"❌ Erro: A aba {e} não foi encontrada no arquivo Excel.")
        return None, None, None
    
def padronizar_fase(valor):
    # Converte para string maiúscula e remove espaços nas pontas
    valor_str = str(valor).upper().strip()
    
    # Regra 1: Se for Alfa, padronizamos para 'Fase Alfa'
    if 'ALFA' in valor_str:
        return 'Fase Alfa'
    
    # Regra 2: Procuramos o primeiro número que aparece no texto (ex: '1A' -> '1', 'FASE 2' -> '2')
    match = re.search(r'\d+', valor_str)
    
    if match:
        numero = int(match.group())
        if numero == 0:
            return 'Fase Alfa'
        return f'Fase {numero}'
    
    # Se não achar número nem Alfa (por segurança), retorna como Nulo ou o valor original
    return valor

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

def preparar_dados_api(df_2022, df_2023, df_2024):
    """
    Filtra os dados e prepara as variáveis, mantendo informações demográficas para a API.
    """
    features_kmeans = [
        'indicador_desempenho_academico',
        'indicador_engajamento',
        'indicador_psicossocial',
        'indicador_autoavaliacao'
    ]
    
    # Adicionamos as colunas que o frontend vai querer exibir
    # Substitua 'NOME' pelo nome exato que estiver na sua planilha (ex: 'NOME_ALUNO')
    colunas_demograficas = ['RA', 'Fase', 'Idade']

    df_2022['origem_aba'] = 2022
    df_2023['origem_aba'] = 2023
    df_2024['origem_aba'] = 2024
    
    df_lista = [df_2022, df_2023, df_2024]
    
    # Unimos as features + demografia para não perder os dados na hora de limpar os nulos
    colunas_disponiveis = colunas_demograficas + features_kmeans + ['origem_aba'] 
    
    for i in range(len(df_lista)):
        # Garante que só vai tentar filtrar as colunas se elas existirem na aba
        cols_to_keep = [c for c in colunas_disponiveis if c in df_lista[i].columns]
        # Aplica o dropna() apenas olhando para as colunas de notas
        df_lista[i] = df_lista[i][cols_to_keep].dropna(subset=features_kmeans)

    df = pd.concat(df_lista, ignore_index=True)
    df = df[df['indicador_autoavaliacao'] != 0].reset_index(drop=True)
    
    # Filtrar dados de 2024
    df_2024 = df[df['origem_aba'] == 2024].copy()
    
    if 'RA' in df_2024.columns:
            df_2024['RA'] = df_2024['RA'].astype(str).str.extract(r'(\d+)')[0]

    # Calcular as dimensões e já arredondar para a API
    df_2024['dimensao_academica'] = df_2024[['indicador_desempenho_academico', 'indicador_engajamento']].mean(axis=1)
    df_2024['dimensao_psicossocial'] = df_2024[['indicador_psicossocial', 'indicador_autoavaliacao']].mean(axis=1)

    return df_2024

def exportar_base_api(df_2024):
    """
    Salva os dados preparados na pasta da API para serem consumidos pelo FastAPI.
    """
    current_directory = Path(__file__).resolve().parent.parent
    pasta_destino = current_directory / 'api' / 'database'
    
    # Cria a pasta database caso ela ainda não exista
    pasta_destino.mkdir(parents=True, exist_ok=True)
    
    caminho_output = pasta_destino / 'alunos_db.csv'
    
    # Selecionamos apenas as colunas vitais para não pesar o servidor
    colunas_finais = [
        'ra', 'fase', 'idade', 'indicador_desempenho_academico', 'indicador_engajamento', 
        'indicador_psicossocial', 'indicador_autoavaliacao', 'dimensao_academica', 'dimensao_psicossocial'
    ]

    cols_to_save = [c for c in colunas_finais if c in df_2024.columns]
    
    df_final = df_2024[cols_to_save]
    df_final.to_csv(caminho_output, index=False, encoding='utf-8')
    
    print(f"✅ Base da API salva com sucesso ({len(df_final)} alunos) em:")
    print(f"📁 {caminho_output}")

def formatar_colunas_lower(df):
    """
    Transforma todos os nomes das colunas em letras minúsculas 
    e remove espaços em branco extras.
    """
    # .str.lower() converte para minúsculo
    # .str.strip() remove espaços acidentais no início ou fim (ex: " NOME")
    # .str.replace(' ', '_') substitui espaços internos por underscores (boa prática)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    return df

if __name__ == "__main__":
    df_22, df_23, df_24 = carregar_dados()
    
    if df_22 is not None:
        df_22, df_23, df_24 = renomear_colunas(df_22, df_23, df_24)
        df_pronto = preparar_dados_api(df_22, df_23, df_24)
        df_pronto = formatar_colunas_lower(df_pronto)
        df_pronto['fase'] = df_pronto['fase'].apply(padronizar_fase)
        exportar_base_api(df_pronto)