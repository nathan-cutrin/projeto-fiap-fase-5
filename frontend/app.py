import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Classificação de Perfil do Aluno")
st.write("Busque um aluno pelo RA para preencher automaticamente os indicadores, ou insira-os manualmente.")

if "acad_slider" not in st.session_state:
    st.session_state["acad_slider"] = 5.0
if "psico_slider" not in st.session_state:
    st.session_state["psico_slider"] = 5.0
if "aluno_info" not in st.session_state:
    st.session_state.aluno_info = None

with st.container(border=True):
    st.subheader("🔍 Buscar Aluno (Base 2024)")
    busca_ra = st.text_input("Digite o RA do Aluno (somente números, ex: 904)")
    
    if st.button("Buscar Aluno"):
        if not busca_ra.strip():
            st.warning("Por favor, digite um número de RA válido.")
        else:
            try:
                ra_limpo = busca_ra.strip()
                resp_busca = requests.get(f"{API_URL}/student/{ra_limpo}")
                
                if resp_busca.status_code == 200:
                    dados_aluno = resp_busca.json()
                    
                    st.session_state["acad_slider"] = float(dados_aluno["dimensao_academica"])
                    st.session_state["psico_slider"] = float(dados_aluno["dimensao_psicossocial"])
                    
                    st.session_state.aluno_info = dados_aluno
                    st.success(f"Aluno com RA {dados_aluno['ra']} encontrado!")
                    st.rerun()
                    
                    st.session_state.aluno_info = dados_aluno
                    
                    st.success(f"Aluno com RA {dados_aluno['ra']} encontrado!")
                    st.rerun()
                else:
                    st.error("Aluno não encontrado na base de dados.")
            except Exception as e:
                st.error(f"Erro de conexão: {e}")

if st.session_state.aluno_info:
    info = st.session_state.aluno_info
    with st.expander("ℹ️ Detalhes e Indicadores Brutos do Aluno", expanded=True):
        
        # Dados Demográficos
        col_demo1, col_demo2 = st.columns(2)
        col_demo1.markdown(f"**Fase:** {info.get('fase')}")
        idade_str = int(info.get('idade')) if info.get('idade') is not None else "N/A"
        col_demo2.markdown(f"**Idade:** {idade_str}")
        
        st.divider()
        
        st.markdown("##### Indicadores Originais")
        ind1, ind2, ind3, ind4 = st.columns(4)
        ind1.metric("Desemp. Acad.", info.get('indicador_desempenho_academico'))
        ind2.metric("Engajamento", info.get('indicador_engajamento'))
        ind3.metric("Psicossocial", info.get('indicador_psicossocial'))
        ind4.metric("Autoavaliação", info.get('indicador_autoavaliacao'))

st.divider()
st.subheader("📊 Dimensões de Agrupamento (Input do Modelo)")
st.caption("Estas dimensões são alimentadas automaticamente ao buscar um aluno. Você também pode ajustá-las para simular cenários.")

col1, col2 = st.columns(2)

with col1:
    dim_acad = st.slider(
        "Dimensão Acadêmica", 
        0.0, 10.0, 
        key="acad_slider", 
        step=0.1
    )

with col2:
    dim_psico = st.slider(
        "Dimensão Psicossocial", 
        0.0, 10.0, 
        key="psico_slider", #
        step=0.1
    )

if st.button("Analisar Perfil", type="primary"):
    payload = {
        "dimensao_academica": dim_acad,
        "dimensao_psicossocial": dim_psico
    }
    
    try:
        resposta = requests.post(f"{API_URL}/predict", json=payload)
        
        if resposta.status_code == 200:
            resultado = resposta.json()
            
            cluster = resultado.get("classe_predita")
            metodo = resultado.get("metodo")
            
            st.success("Análise de Cluster concluída!")
            
            persona_nome = resultado.get("persona_nome")
            persona_desc = resultado.get("persona_descricao")
            if persona_nome:
                st.markdown(f"### 🎯 Perfil Identificado: **{persona_nome}**")
                st.info(persona_desc)
            
            col_a, col_b = st.columns(2)
            col_a.metric(label="Grupo Matemático", value=f"Cluster {cluster}")
            col_b.caption(f"Processado via: {metodo}")
            
        else:
            st.error(f"Erro na API: {resposta.status_code} - {resposta.text}")
            
    except Exception as e:
        st.error(f"Falha de conexão com a API: {e}")