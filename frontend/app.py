import streamlit as st
import requests

st.title("Previsão de Risco do Aluno")

faltas = st.number_input("Número de Faltas", min_value=0, step=1)

if st.button("Calcular Risco"):
    url = "https://sua-api-aqui.onrender.com/predict"
    
    dados = {
        "faltas": faltas
    }
    
    resposta = requests.post(url, json=dados)
    
    if resposta.status_code == 200:
        resultado = resposta.json()
        st.write(resultado)
    else:
        st.error("Erro ao conectar com a API.")