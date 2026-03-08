def converter_stats(json_response: dict) -> dict:
    return {int(k): v for k, v in json_response.items()}

def montar_rows_stats(stats: dict) -> list:
    colunas_labels = {
        "indicador_desempenho_academico": "Desemp. Acadêmico",
        "indicador_engajamento":          "Engajamento",
        "indicador_psicossocial":         "Psicossocial",
        "indicador_autoavaliacao":        "Autoavaliação",
        "dimensao_academica":             "Dim. Acadêmica",
        "dimensao_psicossocial":          "Dim. Psicossocial",
    }
    rows = []
    for col_key, label in colunas_labels.items():
        if col_key in stats:
            s = stats[col_key]
            rows.append({
                "Indicador": label,
                "Média":     s["mean"],
                "Mediana":   s["median"],
                "Mín":       s["min"],
                "Máx":       s["max"],
            })
    return rows