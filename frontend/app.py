import streamlit as st
import requests
import os

st.title("Classificação de Perfil do Aluno")
st.write("Insira os indicadores do aluno para descobrir em qual grupo de acompanhamento da ONG ele se encaixa.")

# 1. Entradas do usuário baseadas nas features do K-Means
col1, col2 = st.columns(2)

with col1:
    dimensao_psicossocial = st.slider("Dimensão Psicossocial", min_value=0.0, max_value=10.0, step=0.1, value=5.0)

with col2:
    dimensao_academica = st.slider("Dimensão Acadêmica", min_value=0.0, max_value=10.0, step=0.1, value=5.0)


if st.button("Analisar Perfil"):
    api_url = os.getenv("API_URL", "http://localhost:8000/predict")
    
    # 2. O payload precisa bater EXATAMENTE com os atributos da sua classe StudentData
    # Lembre-se de verificar no seu schemas.py quais são os nomes exatos esperados
    dados = {
        "dimensao_psicossocial": dimensao_psicossocial,
        "dimensao_academica": dimensao_academica,
    }
    
    try:
        resposta = requests.post(api_url, json=dados)
        
        if resposta.status_code == 200:
            resultado = resposta.json()
            
            # 3. Resgatando o Cluster e o Método
            # A API devolve 'risco_predito', mas agora sabemos que isso é o Cluster ID
            cluster = resultado.get("classe_predita")
            metodo = resultado.get("metodo")
            
            st.success("Análise concluída com sucesso!")
            
            # Exibição visual do resultado
            st.metric(label="Grupo de Perfil (Cluster)", value=f"Cluster {cluster}")
            st.caption(f"Processado via: {metodo}")
            
        else:
            st.error(f"Erro retornado pela API: {resposta.status_code}")
            st.write(resposta.text)
            
    except Exception as e:
        st.error(f"Falha de conexão com a API: {e}")