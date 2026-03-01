import streamlit as st
import requests
import os

st.title("Previsão de Risco do Aluno")

faltas = st.number_input("Número de Faltas", min_value=0, step=1)

if st.button("Calcular Risco"):
    api_url = os.getenv("API_URL", "http://localhost:8000/predict")
    
    dados = {
        "faltas": faltas
    }
    
    try:
        resposta = requests.post(api_url, json=dados)
        
        if resposta.status_code == 200:
            resultado = resposta.json()
            st.write(resultado)
        else:
            st.error(f"Erro retornado pela API: {resposta.status_code}")
    except Exception as e:
        st.error(f"Falha de conexão com a API: {e}")