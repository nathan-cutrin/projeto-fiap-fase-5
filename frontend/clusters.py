import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ─────────────────────────────────────────────
# Personas dos clusters
# ─────────────────────────────────────────────

PERSONAS = {
    1: {
        "nome":  "Resiliente Emocional",
        "emoji": "💛",
        "cor":   "#FFD700",
        "descricao": (
            "Aluno com forte desenvolvimento psicossocial, mas que ainda não converteu "
            "esse potencial em desempenho acadêmico. Demonstra engajamento emocional e "
            "autoavaliação positiva. Com suporte pedagógico direcionado, tem grande "
            "potencial de evolução acadêmica."
        ),
        "acoes": [
            "Reforço em conteúdos acadêmicos",
            "Mentoria individualizada",
            "Atividades que conectem o engajamento emocional ao aprendizado",
        ],
    },
    2: {
        "nome":  "Alto Desempenho",
        "emoji": "🌟",
        "cor":   "#00E676",
        "descricao": (
            "Aluno com desempenho elevado em ambas as dimensões — acadêmica e psicossocial. "
            "Representa o perfil mais desenvolvido da base. Está bem preparado para "
            "avançar de fase e pode atuar como referência positiva para outros alunos."
        ),
        "acoes": [
            "Programas de aceleração e desafios avançados",
            "Atividades de liderança e protagonismo",
            "Manutenção do acompanhamento para sustentação do desempenho",
        ],
    },
    3: {
        "nome":  "Atenção Prioritária",
        "emoji": "🔴",
        "cor":   "#FF4444",
        "descricao": (
            "Aluno com baixo desempenho nas duas dimensões — acadêmica e psicossocial. "
            "Representa o maior risco de defasagem escolar. Necessita de atenção "
            "imediata tanto da equipe pedagógica quanto do suporte psicossocial."
        ),
        "acoes": [
            "Acompanhamento psicopedagógico urgente",
            "Plano de recuperação acadêmica personalizado",
            "Avaliação de fatores externos (família, saúde, vulnerabilidade social)",
        ],
    },
    4: {
        "nome":  "Destaque Acadêmico",
        "emoji": "📘",
        "cor":   "#29B6F6",
        "descricao": (
            "Aluno com excelente desempenho acadêmico e engajamento, mas com dimensão "
            "psicossocial em nível intermediário. Pode estar sobrecarregado ou com menor "
            "suporte emocional. Merece atenção no equilíbrio entre performance e bem-estar."
        ),
        "acoes": [
            "Acompanhamento do bem-estar emocional",
            "Atividades de equilíbrio socioemocional",
            "Manter o estímulo acadêmico sem gerar sobrecarga",
        ],
    },
}

# Centroides no espaço original (calculados a partir dos scaled + stats de treino)
CENTROIDES = {
    1: {"dimensao_academica": 5.292, "dimensao_psicossocial": 7.911},
    2: {"dimensao_academica": 8.241, "dimensao_psicossocial": 8.382},
    3: {"dimensao_academica": 6.027, "dimensao_psicossocial": 5.888},
    4: {"dimensao_academica": 8.240, "dimensao_psicossocial": 7.206},
}

REF_STATS = {
    "dimensao_academica":    {"mean": 7.2614, "std": 1.6950},
    "dimensao_psicossocial": {"mean": 7.7672, "std": 0.8570},
}

N_POR_CLUSTER = {1: 281, 2: 418, 3: 83, 4: 252}


@st.cache_data(ttl=3600)
def carregar_stats_api():
    try:
        resp = requests.get(f"{API_URL}/clusters/stats")
        if resp.status_code == 200:
            return {int(k): v for k, v in resp.json().items()}
    except Exception:
        pass
    return {}


def gerar_pontos_cluster(cluster_id, n, stats_api):
    """Gera pontos sintéticos representativos do cluster."""
    np.random.seed(cluster_id * 10)
    cx = CENTROIDES[cluster_id]["dimensao_academica"]
    cy = CENTROIDES[cluster_id]["dimensao_psicossocial"]

    if stats_api and cluster_id in stats_api:
        s = stats_api[cluster_id]
        sx = (s["dimensao_academica"]["max"] - s["dimensao_academica"]["min"]) / 6
        sy = (s["dimensao_psicossocial"]["max"] - s["dimensao_psicossocial"]["min"]) / 6
    else:
        sx = REF_STATS["dimensao_academica"]["std"] * 0.6
        sy = REF_STATS["dimensao_psicossocial"]["std"] * 0.6

    x = np.clip(np.random.normal(cx, sx, n), 0, 10)
    y = np.clip(np.random.normal(cy, sy, n), 0, 10)
    return x, y


def render():
    st.title("🗂️ Caracterização dos Clusters")
    st.write(
        "Perfil dos 4 grupos de alunos identificados pelo modelo KMeans, "
        "com base nas dimensões acadêmica e psicossocial."
    )

    stats_api = carregar_stats_api()

    # ─────────────────────────────────────────
    # Cards de personas
    # ─────────────────────────────────────────
    st.subheader("👤 Perfis Identificados")
    cols = st.columns(4)
    for i, (cluster_id, p) in enumerate(PERSONAS.items()):
        n = N_POR_CLUSTER[cluster_id]
        pct = round(n / sum(N_POR_CLUSTER.values()) * 100, 1)
        with cols[i]:
            st.markdown(
                f"""
                <div style="border: 2px solid {p['cor']}; border-radius: 10px;
                            padding: 16px; text-align: center; min-height: 160px;">
                    <div style="font-size: 2rem;">{p['emoji']}</div>
                    <div style="font-weight: bold; font-size: 1rem; margin: 6px 0;">
                        Cluster {cluster_id}
                    </div>
                    <div style="color: {p['cor']}; font-weight: bold;">{p['nome']}</div>
                    <div style="font-size: 0.85rem; margin-top: 6px; color: #888;">
                        {n} alunos ({pct}%)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ─────────────────────────────────────────
    # Gráfico de dispersão
    # ─────────────────────────────────────────
    st.subheader("📍 Dispersão dos Clusters")
    st.caption("Pontos sintéticos representativos gerados a partir dos centroides e estatísticas reais de cada cluster.")

    fig = go.Figure()

    for cluster_id, p in PERSONAS.items():
        n = N_POR_CLUSTER[cluster_id]
        x, y = gerar_pontos_cluster(cluster_id, n, stats_api)

        # Pontos do cluster
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            name=f"Cluster {cluster_id} — {p['nome']}",
            marker=dict(color=p["cor"], size=7, opacity=0.55),
        ))

        # Centroide
        cx = CENTROIDES[cluster_id]["dimensao_academica"]
        cy = CENTROIDES[cluster_id]["dimensao_psicossocial"]
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode="markers+text",
            name=f"Centroide {cluster_id}",
            text=[f"  C{cluster_id}"],
            textposition="middle right",
            textfont=dict(size=13, color=p["cor"]),
            marker=dict(
                color=p["cor"], size=16,
                symbol="x", line=dict(width=2, color="white")
            ),
            showlegend=False,
        ))

        # Mediana
        if stats_api and cluster_id in stats_api:
            mx = stats_api[cluster_id]["dimensao_academica"]["median"]
            my = stats_api[cluster_id]["dimensao_psicossocial"]["median"]
            fig.add_trace(go.Scatter(
                x=[mx], y=[my],
                mode="markers",
                name=f"Mediana {cluster_id}",
                marker=dict(
                    color=p["cor"], size=12,
                    symbol="diamond", line=dict(width=2, color="white")
                ),
                showlegend=False,
            ))

    # Anotações de legenda dos símbolos
    fig.add_annotation(
        x=0.01, y=0.99, xref="paper", yref="paper",
        text="✖ Centroide   ◆ Mediana",
        showarrow=False, font=dict(size=11, color="#aaa"),
        align="left"
    )

    fig.update_layout(
        xaxis_title="Dimensão Acadêmica",
        yaxis_title="Dimensão Psicossocial",
        height=620,
        width=620,
        dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(range=[0, 10.5], constrain="domain"),
        yaxis=dict(range=[0, 10.5], scaleanchor="x", scaleratio=1),
    )

    _, col_chart, _ = st.columns([1, 2, 1])
    with col_chart:
        st.plotly_chart(fig, use_container_width=False)

    st.divider()

    # ─────────────────────────────────────────
    # Gráfico de barras comparativo
    # ─────────────────────────────────────────
    st.subheader("📊 Comparativo entre Clusters")

    if stats_api:
        colunas_plot = {
            "indicador_desempenho_academico": "Desemp. Acadêmico",
            "indicador_engajamento":          "Engajamento",
            "indicador_psicossocial":         "Psicossocial",
            "indicador_autoavaliacao":        "Autoavaliação",
            "dimensao_academica":             "Dim. Acadêmica",
            "dimensao_psicossocial":          "Dim. Psicossocial",
        }

        metrica = st.selectbox(
            "Selecione o indicador:",
            options=list(colunas_plot.keys()),
            format_func=lambda x: colunas_plot[x],
        )

        clusters_labels = [f"Cluster {c}\n{PERSONAS[c]['nome']}" for c in [1, 2, 3, 4]]
        medias = [
            stats_api[c][metrica]["mean"] if c in stats_api and metrica in stats_api[c] else 0
            for c in [1, 2, 3, 4]
        ]
        medianas = [
            stats_api[c][metrica]["median"] if c in stats_api and metrica in stats_api[c] else 0
            for c in [1, 2, 3, 4]
        ]
        cores = [PERSONAS[c]["cor"] for c in [1, 2, 3, 4]]

        fig_bar = go.Figure(data=[
            go.Bar(
                name="Média",
                x=clusters_labels,
                y=medias,
                marker_color=cores,
                text=[f"{v:.2f}" for v in medias],
                textposition="outside",
                opacity=0.85,
            ),
            go.Bar(
                name="Mediana",
                x=clusters_labels,
                y=medianas,
                marker_color=cores,
                text=[f"{v:.2f}" for v in medianas],
                textposition="outside",
                opacity=0.45,
            ),
        ])
        fig_bar.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 11], title="Valor"),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("API indisponível — não foi possível carregar as estatísticas dos clusters.")

    st.divider()

    # ─────────────────────────────────────────
    # Tabela de stats
    # ─────────────────────────────────────────
    st.subheader("📋 Estatísticas por Cluster")

    if stats_api:
        colunas_tabela = {
            "dimensao_academica":             "Dim. Acadêmica",
            "dimensao_psicossocial":          "Dim. Psicossocial",
            "indicador_desempenho_academico": "Desemp. Acadêmico",
            "indicador_engajamento":          "Engajamento",
            "indicador_psicossocial":         "Psicossocial",
            "indicador_autoavaliacao":        "Autoavaliação",
        }

        cluster_sel = st.radio(
            "Cluster:",
            options=[1, 2, 3, 4],
            format_func=lambda c: f"Cluster {c} — {PERSONAS[c]['emoji']} {PERSONAS[c]['nome']}",
            horizontal=True,
        )

        if cluster_sel in stats_api:
            s = stats_api[cluster_sel]
            rows = []
            for col_key, label in colunas_tabela.items():
                if col_key in s:
                    rows.append({
                        "Indicador": label,
                        "Média":     s[col_key]["mean"],
                        "Mediana":   s[col_key]["median"],
                        "Mín":       s[col_key]["min"],
                        "Máx":       s[col_key]["max"],
                    })
            df_table = pd.DataFrame(rows).set_index("Indicador")
            st.dataframe(df_table, use_container_width=True)
            st.caption(f"Total de alunos neste cluster: **{s.get('n_alunos', '?')}**")

    st.divider()

    # ─────────────────────────────────────────
    # Descrições detalhadas
    # ─────────────────────────────────────────
    st.subheader("📖 Descrição e Ações Recomendadas")

    for cluster_id, p in PERSONAS.items():
        with st.expander(f"{p['emoji']} Cluster {cluster_id} — {p['nome']}", expanded=False):
            st.markdown(p["descricao"])
            st.markdown("**Ações recomendadas:**")
            for acao in p["acoes"]:
                st.markdown(f"- {acao}")