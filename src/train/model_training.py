from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def treinar_modelo(X_scaled):
    """
    Função para treinar o modelo de clusterização hierárquica.
    """
    modelo_hc = AgglomerativeClustering(n_clusters=4, metric='cosine', linkage='average')
    modelo_hc.fit(X_scaled)
    
    # Avaliação do modelo (Silhouette Score)
    score = silhouette_score(X_scaled, modelo_hc.labels_)
    print(f"Silhouette Score: {score}")
    
    return modelo_hc, score

def avaliar_modelo(X_scaled, modelo_hc):
    """
    Função para avaliar a qualidade do modelo usando uma métrica.
    """
    score = silhouette_score(X_scaled, modelo_hc.labels_)
    print(f"Silhouette Score: {score}")
    return score