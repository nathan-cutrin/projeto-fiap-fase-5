import streamlit as st
import requests
import pandas as pd
import os
from utils import converter_stats, montar_rows_stats
from monitoring import render as render_monitoring

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(layout="wide")

st.title("Classificação de Perfil do Aluno")

aba1, aba2 = st.tabs(["🎓 Classificar Aluno", "📡 Monitoramento de Drift"])

# ─────────────────────────────────────────────
# ABA 1 — Classificação
# ─────────────────────────────────────────────

with aba1:
    st.write("Busque um aluno pelo RA para preencher automaticamente os indicadores, ou insira-os manualmente.")

    # --- Inicialização do estado ---
    for key in ["ind_desempenho", "ind_engajamento", "ind_psicossocial", "ind_autoavaliacao"]:
        if key not in st.session_state:
            st.session_state[key] = 0.0
        if f"{key}_input" not in st.session_state:
            st.session_state[f"{key}_input"] = 0.0

    if "aluno_info" not in st.session_state:
        st.session_state.aluno_info = None

    if "campos_ausentes_busca" not in st.session_state:
        st.session_state.campos_ausentes_busca = []

    # --- Carrega stats dos clusters uma única vez por sessão ---
    @st.cache_data(ttl=3600)
    def carregar_stats_clusters():
        try:
            resp = requests.get(f"{API_URL}/clusters/stats")
            if resp.status_code == 200:
                return converter_stats(resp.json())
        except Exception:
            pass
        return {}

    def exibir_stats_cluster(cluster_id: int, stats_clusters: dict):
        stats = stats_clusters.get(cluster_id)
        if not stats:
            st.warning("Estatísticas do cluster não disponíveis.")
            return

        n = stats.get("n_alunos", "?")
        st.markdown(f"#### 📊 Comparativo com o Cluster {cluster_id} — {n} alunos na base")

        rows = montar_rows_stats(stats)
        df_stats = pd.DataFrame(rows).set_index("Indicador")
        st.dataframe(df_stats, use_container_width=True)

    # --- Busca por RA ---
    with st.container(border=True):
        st.subheader("🔍 Buscar Aluno (Base 2024)")
        busca_ra = st.text_input("Digite o RA do Aluno (somente números, ex: 904)")

        if st.button("Buscar Aluno"):
            if not busca_ra.strip():
                st.warning("Por favor, digite um número de RA válido.")
            else:
                try:
                    resp_busca = requests.get(f"{API_URL}/student/{busca_ra.strip()}")

                    if resp_busca.status_code == 200:
                        dados_aluno = resp_busca.json()
                        st.session_state.aluno_info = dados_aluno

                        mapa_campos = {
                            "ind_desempenho":    "indicador_desempenho_academico",
                            "ind_engajamento":   "indicador_engajamento",
                            "ind_psicossocial":  "indicador_psicossocial",
                            "ind_autoavaliacao": "indicador_autoavaliacao",
                        }

                        ausentes = []
                        for key, campo_api in mapa_campos.items():
                            valor = dados_aluno.get(campo_api)
                            v = float(valor) if valor is not None else 0.0
                            st.session_state[key] = v
                            st.session_state[f"{key}_input"] = v
                            if valor is None:
                                ausentes.append(key)

                        st.session_state.campos_ausentes_busca = ausentes
                        st.success(f"Aluno com RA {dados_aluno['ra']} encontrado!")
                        st.rerun()
                    else:
                        st.error("Aluno não encontrado na base de dados.")
                except Exception as e:
                    st.error(f"Erro de conexão: {e}")

    # --- Detalhes do aluno encontrado ---
    if st.session_state.aluno_info:
        info = st.session_state.aluno_info
        with st.expander("ℹ️ Dados do Aluno", expanded=True):
            col1, col2 = st.columns(2)
            col1.markdown(f"**Fase:** {info.get('fase')}")
            idade_str = int(info.get('idade')) if info.get('idade') is not None else "N/A"
            col2.markdown(f"**Idade:** {idade_str}")

    st.divider()

    # --- Inputs dos 4 indicadores ---
    st.subheader("📋 Indicadores do Aluno")
    st.caption("Preenchidos automaticamente ao buscar um aluno. Edite os campos que estiverem faltando.")

    def indicador_input(label, key, col):
        slider_key = key
        input_key  = f"{key}_input"

        def slider_mudou():
            st.session_state[input_key] = st.session_state[slider_key]

        def input_mudou():
            st.session_state[slider_key] = st.session_state[input_key]

        ausente = key in st.session_state.campos_ausentes_busca
        if ausente:
            col.markdown(f"**{label}** 🔴 *não encontrado*")
        else:
            col.markdown(f"**{label}**")

        num_col, slider_col = col.columns([1, 4])

        slider_col.slider(
            label=label,
            min_value=0.0,
            max_value=10.0,
            step=0.01,
            key=slider_key,
            on_change=slider_mudou,
            label_visibility="collapsed",
        )

        num_col.number_input(
            label=label,
            min_value=0.0,
            max_value=10.0,
            step=0.01,
            format="%.2f",
            key=input_key,
            on_change=input_mudou,
            label_visibility="collapsed",
        )

        return st.session_state[slider_key]

    col1, col2, col3, col4 = st.columns(4)

    ind_desempenho    = indicador_input("🎓 Desemp. Acadêmico", "ind_desempenho",    col1)
    ind_engajamento   = indicador_input("💡 Engajamento",        "ind_engajamento",   col2)
    ind_psicossocial  = indicador_input("🧠 Psicossocial",       "ind_psicossocial",  col3)
    ind_autoavaliacao = indicador_input("🪞 Autoavaliação",      "ind_autoavaliacao", col4)

    # --- Cálculo interno das dimensões ---
    dim_academica    = round((ind_desempenho  + ind_engajamento)  / 2, 4)
    dim_psicossocial = round((ind_psicossocial + ind_autoavaliacao) / 2, 4)

    # --- Preview das dimensões calculadas ---
    with st.container(border=True):
        st.caption("📐 Dimensões calculadas automaticamente (input do modelo)")
        c1, c2 = st.columns(2)
        c1.metric("Dimensão Acadêmica",    f"{dim_academica:.2f}",    help="Média entre Desempenho Acadêmico e Engajamento")
        c2.metric("Dimensão Psicossocial", f"{dim_psicossocial:.2f}", help="Média entre Psicossocial e Autoavaliação")

    st.divider()

    # --- Aviso de campos faltando ---
    if st.session_state.campos_ausentes_busca:
        st.warning(f"⚠️ {len(st.session_state.campos_ausentes_busca)} indicador(es) não encontrado(s) para este aluno. Ajuste os campos 🔴 antes de analisar.")

    # --- Botão de análise ---
    if st.button("Analisar Perfil", type="primary"):
        payload = {
            "dimensao_academica":    dim_academica,
            "dimensao_psicossocial": dim_psicossocial,
        }

        try:
            resposta = requests.post(f"{API_URL}/predict", json=payload)

            if resposta.status_code == 200:
                resultado = resposta.json()

                cluster      = resultado.get("classe_predita")
                metodo       = resultado.get("metodo")
                persona_nome = resultado.get("persona_nome")
                persona_desc = resultado.get("persona_descricao")

                st.success("Análise de Cluster concluída!")

                if persona_nome:
                    st.markdown(f"### 🎯 Perfil Identificado: **{persona_nome}**")
                    st.info(persona_desc)

                col_a, col_b = st.columns(2)
                col_a.metric(label="Grupo Matemático", value=f"Cluster {cluster}")
                col_b.caption(f"Processado via: {metodo}")

                st.divider()

                stats_clusters = carregar_stats_clusters()
                exibir_stats_cluster(cluster, stats_clusters)

            else:
                st.error(f"Erro na API: {resposta.status_code} - {resposta.text}")

        except Exception as e:
            st.error(f"Falha de conexão com a API: {e}")


# ─────────────────────────────────────────────
# ABA 2 — Monitoramento de Drift
# ─────────────────────────────────────────────

with aba2:
    render_monitoring()