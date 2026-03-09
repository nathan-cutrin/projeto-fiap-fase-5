import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# Estatísticas de referência — dados de treino 2024
# ─────────────────────────────────────────────

REF = {
    "dimensao_academica": {
        "mean": 7.2614,
        "std":  1.6950,
        "min":  0.0,
        "max":  10.0,
        "q25":  6.2500,
        "q50":  7.5244,
        "q75":  8.5471,
    },
    "dimensao_psicossocial": {
        "mean": 7.7672,
        "std":  0.8570,
        "min":  3.547,
        "max":  10.001,
        "q25":  7.3810,
        "q50":  7.9220,
        "q75":  8.3390,
    },
}

REF_CLUSTERS = {1: 27.2, 2: 40.4, 3: 8.0, 4: 24.4}
N_TREINO = 1034
DRIFT_PVALUE_THRESHOLD = 0.05
DRIFT_CLUSTER_THRESHOLD = 10.0  # pp


def gerar_dados_simulados(n, delta_acad, delta_psico, ruido_extra):
    """Gera amostra sintética deslocada em relação à referência."""
    np.random.seed(42)
    acad  = np.clip(np.random.normal(
        REF["dimensao_academica"]["mean"]    + delta_acad,
        REF["dimensao_academica"]["std"]     + ruido_extra,
        n
    ), 0, 10)
    psico = np.clip(np.random.normal(
        REF["dimensao_psicossocial"]["mean"] + delta_psico,
        REF["dimensao_psicossocial"]["std"]  + ruido_extra,
        n
    ), 0, 10)
    return pd.DataFrame({"dimensao_academica": acad, "dimensao_psicossocial": psico})


def classificar_clusters_simulados(df):
    """
    Classifica clusters simulados usando os centroides reais do modelo
    (no espaço padronizado) sem precisar carregar o joblib aqui.
    """
    centroides_scaled = np.array([
        [-1.16187482,  0.16824004],
        [ 0.57773654,  0.71738231],
        [-0.72842262, -2.19296319],
        [ 0.57719059, -0.65525918],
    ])

    mean_ref = np.array([REF["dimensao_academica"]["mean"], REF["dimensao_psicossocial"]["mean"]])
    std_ref  = np.array([REF["dimensao_academica"]["std"],  REF["dimensao_psicossocial"]["std"]])

    X = (df[["dimensao_academica", "dimensao_psicossocial"]].values - mean_ref) / std_ref
    distancias = np.linalg.norm(X[:, np.newaxis] - centroides_scaled, axis=2)
    return distancias.argmin(axis=1) + 1  # offset +1


def badge_pvalue(pvalue):
    if pvalue < DRIFT_PVALUE_THRESHOLD:
        return f"🔴 p = {pvalue:.4f} — **Drift detectado**"
    else:
        return f"✅ p = {pvalue:.4f} — Sem drift"


def badge_cluster(desvio):
    if abs(desvio) > DRIFT_CLUSTER_THRESHOLD:
        return "🔴"
    elif abs(desvio) > 5:
        return "🟡"
    else:
        return "✅"


# ─────────────────────────────────────────────
# Layout principal
# ─────────────────────────────────────────────

def render():
    st.title("📡 Monitoramento de Drift do Modelo")
    st.write(
        "Acompanhe a estabilidade do modelo comparando a distribuição de referência "
        "(dados de treino 2024) com dados simulados de produção. "
        "Em produção real, os dados de entrada da API substituiriam a simulação."
    )

    st.info(
        "**Como funciona:** use os controles abaixo para simular cenários onde os dados "
        "de produção começam a divergir do treino. O sistema aplica o teste KS e monitora "
        "a distribuição de clusters para emitir alertas automáticos.",
        icon="ℹ️"
    )

    # ─────────────────────────────────────────
    # Linha de Base
    # ─────────────────────────────────────────
    with st.expander("📊 Linha de Base — Dados de Treino 2024", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de alunos", f"{N_TREINO:,}")
        col2.metric("Dim. Acadêmica (média)", f"{REF['dimensao_academica']['mean']:.2f}")
        col3.metric("Dim. Psicossocial (média)", f"{REF['dimensao_psicossocial']['mean']:.2f}")
        col4.metric("Clusters", "4")

        st.markdown("##### Distribuição de Clusters (Referência)")
        df_ref_clusters = pd.DataFrame({
            "Cluster": [f"Cluster {k}" for k in REF_CLUSTERS],
            "Proporção (%)": list(REF_CLUSTERS.values()),
        })
        fig_ref = px.bar(
            df_ref_clusters, x="Cluster", y="Proporção (%)",
            color="Cluster",
            color_discrete_sequence=["#4C9BE8", "#5BC98A", "#F4A460", "#E87070"],
            text_auto=".1f"
        )
        fig_ref.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_ref, use_container_width=True)

        st.markdown("##### Estatísticas das Features")
        df_stats = pd.DataFrame({
            "Feature": ["Dim. Acadêmica", "Dim. Psicossocial"],
            "Média":   [REF["dimensao_academica"]["mean"],   REF["dimensao_psicossocial"]["mean"]],
            "Desvio Padrão": [REF["dimensao_academica"]["std"], REF["dimensao_psicossocial"]["std"]],
            "Mín":     [REF["dimensao_academica"]["min"],    REF["dimensao_psicossocial"]["min"]],
            "Mediana": [REF["dimensao_academica"]["q50"],    REF["dimensao_psicossocial"]["q50"]],
            "Máx":     [REF["dimensao_academica"]["max"],    REF["dimensao_psicossocial"]["max"]],
        }).set_index("Feature").round(4)
        st.dataframe(df_stats, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────
    # Controles do simulador
    # ─────────────────────────────────────────
    st.subheader("🎛️ Simulador de Produção")
    st.caption("Simule como o modelo reagiria a diferentes distribuições de dados em produção.")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        n_simulado = st.slider(
            "Nº de amostras simuladas",
            min_value=50, max_value=1000, value=300, step=50
        )
    with col_s2:
        delta_acad = st.slider(
            "Deslocamento — Dim. Acadêmica",
            min_value=-4.0, max_value=4.0, value=0.0, step=0.1,
            help="Desloca a média da dimensão acadêmica em relação ao treino"
        )
    with col_s3:
        delta_psico = st.slider(
            "Deslocamento — Dim. Psicossocial",
            min_value=-4.0, max_value=4.0, value=0.0, step=0.1,
            help="Desloca a média da dimensão psicossocial em relação ao treino"
        )
    with col_s4:
        ruido_extra = st.slider(
            "Ruído adicional (σ)",
            min_value=0.0, max_value=3.0, value=0.0, step=0.1,
            help="Aumenta o desvio padrão dos dados simulados"
        )

    # ─────────────────────────────────────────
    # Geração dos dados
    # ─────────────────────────────────────────
    np.random.seed(42)
    ref_acad  = np.clip(np.random.normal(REF["dimensao_academica"]["mean"],    REF["dimensao_academica"]["std"],    N_TREINO), 0, 10)
    ref_psico = np.clip(np.random.normal(REF["dimensao_psicossocial"]["mean"], REF["dimensao_psicossocial"]["std"], N_TREINO), 0, 10)

    df_sim = gerar_dados_simulados(n_simulado, delta_acad, delta_psico, ruido_extra)
    clusters_sim = classificar_clusters_simulados(df_sim)

    # KS-Tests
    ks_acad  = stats.ks_2samp(ref_acad,  df_sim["dimensao_academica"].values)
    ks_psico = stats.ks_2samp(ref_psico, df_sim["dimensao_psicossocial"].values)

    # Distribuição de clusters simulados
    unique, counts = np.unique(clusters_sim, return_counts=True)
    dist_sim = {int(c): round(n / len(clusters_sim) * 100, 1) for c, n in zip(unique, counts)}
    for c in [1, 2, 3, 4]:
        if c not in dist_sim:
            dist_sim[c] = 0.0

    st.divider()

    # ─────────────────────────────────────────
    # Alertas KS-Test
    # ─────────────────────────────────────────
    st.subheader("🧪 Resultado do KS-Test (Data Drift)")
    st.caption(f"Threshold: p-value < {DRIFT_PVALUE_THRESHOLD} indica drift estatisticamente significativo.")

    col_ks1, col_ks2 = st.columns(2)
    with col_ks1:
        st.markdown("**Dimensão Acadêmica**")
        st.markdown(badge_pvalue(ks_acad.pvalue))
        st.metric(
            "Estatística KS",
            f"{ks_acad.statistic:.4f}",
            delta=f"Δ média: {delta_acad:+.2f}",
            delta_color="inverse"
        )
    with col_ks2:
        st.markdown("**Dimensão Psicossocial**")
        st.markdown(badge_pvalue(ks_psico.pvalue))
        st.metric(
            "Estatística KS",
            f"{ks_psico.statistic:.4f}",
            delta=f"Δ média: {delta_psico:+.2f}",
            delta_color="inverse"
        )

    st.divider()

    # ─────────────────────────────────────────
    # Histogramas sobrepostos
    # ─────────────────────────────────────────
    st.subheader("📈 Distribuição das Features — Referência vs. Produção Simulada")

    fig_hist = make_subplots(rows=1, cols=2, subplot_titles=[
        "Dimensão Acadêmica", "Dimensão Psicossocial"
    ])

    fig_hist.add_trace(go.Histogram(
        x=ref_acad, name="Referência (treino)", opacity=0.6,
        marker_color="#4C9BE8", nbinsx=30,
        histnorm="probability density"
    ), row=1, col=1)
    fig_hist.add_trace(go.Histogram(
        x=df_sim["dimensao_academica"], name="Produção simulada", opacity=0.6,
        marker_color="#F4A460", nbinsx=30,
        histnorm="probability density"
    ), row=1, col=1)

    fig_hist.add_trace(go.Histogram(
        x=ref_psico, name="Referência (treino)", opacity=0.6,
        marker_color="#4C9BE8", nbinsx=30,
        histnorm="probability density", showlegend=False
    ), row=1, col=2)
    fig_hist.add_trace(go.Histogram(
        x=df_sim["dimensao_psicossocial"], name="Produção simulada", opacity=0.6,
        marker_color="#F4A460", nbinsx=30,
        histnorm="probability density", showlegend=False
    ), row=1, col=2)

    fig_hist.update_layout(
        barmode="overlay", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────
    # Distribuição de clusters
    # ─────────────────────────────────────────
    st.subheader("🗂️ Drift de Clusters — Referência vs. Produção Simulada")
    st.caption(f"Alerta 🔴 se desvio > {DRIFT_CLUSTER_THRESHOLD}pp, 🟡 se > 5pp.")

    cluster_labels = [f"Cluster {c}" for c in [1, 2, 3, 4]]
    ref_vals = [REF_CLUSTERS[c] for c in [1, 2, 3, 4]]
    sim_vals = [dist_sim[c] for c in [1, 2, 3, 4]]

    fig_clusters = go.Figure(data=[
        go.Bar(name="Referência (treino)",    x=cluster_labels, y=ref_vals,
               marker_color="#4C9BE8", text=[f"{v:.1f}%" for v in ref_vals], textposition="outside"),
        go.Bar(name="Produção simulada", x=cluster_labels, y=sim_vals,
               marker_color="#F4A460", text=[f"{v:.1f}%" for v in sim_vals], textposition="outside"),
    ])
    fig_clusters.update_layout(barmode="group", height=380, yaxis_title="Proporção (%)")
    st.plotly_chart(fig_clusters, use_container_width=True)

    # Tabela de desvios
    rows_desvio = []
    for c in [1, 2, 3, 4]:
        ref_v = REF_CLUSTERS[c]
        sim_v = dist_sim[c]
        desvio = sim_v - ref_v
        rows_desvio.append({
            "Cluster":              f"Cluster {c}",
            "Referência (%)":       f"{ref_v:.1f}%",
            "Produção simulada (%)": f"{sim_v:.1f}%",
            "Desvio (pp)":          f"{desvio:+.1f}",
            "Status":               badge_cluster(desvio),
        })
    st.dataframe(pd.DataFrame(rows_desvio).set_index("Cluster"), use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────
    # Resumo executivo
    # ─────────────────────────────────────────
    st.subheader("📋 Resumo do Monitoramento")

    drift_features = ks_acad.pvalue < DRIFT_PVALUE_THRESHOLD or ks_psico.pvalue < DRIFT_PVALUE_THRESHOLD
    drift_clusters = any(abs(dist_sim[c] - REF_CLUSTERS[c]) > DRIFT_CLUSTER_THRESHOLD for c in [1, 2, 3, 4])

    if drift_features and drift_clusters:
        st.error("🔴 **Drift crítico detectado** — features e clusters apresentam divergência significativa. Reavaliação do modelo recomendada.")
    elif drift_features:
        st.warning("🟡 **Drift de features detectado** — os dados de entrada divergem do treino. Monitore a evolução.")
    elif drift_clusters:
        st.warning("🟡 **Drift de clusters detectado** — a distribuição de perfis mudou. Verifique se reflete mudança real na população.")
    else:
        st.success("✅ **Modelo estável** — nenhum drift significativo detectado nas condições simuladas.")

    with st.expander("ℹ️ Como interpretar estes resultados"):
        st.markdown("""
        **KS-Test (Kolmogorov-Smirnov):**
        Compara as distribuições estatísticas das features entre treino e produção.
        Um p-value abaixo de 0.05 indica que as distribuições são significativamente diferentes — sinal de que os dados de entrada mudaram.

        **Drift de Clusters:**
        Monitora se a proporção de alunos por perfil se mantém estável.
        Um desvio grande pode indicar mudança real na população ou degradação do modelo.

        **Em produção real:**
        Os dados simulados seriam substituídos pelos inputs reais recebidos pelo endpoint `/predict` ao longo do tempo,
        permitindo monitoramento contínuo e automático da saúde do modelo.
        """)