from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def treinar_modelo(X_scaled, k=4):
    """
    Treina o modelo K-Means e retorna o modelo ajustado.
    """
    modelo_kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    modelo_kmeans.fit(X_scaled)
    return modelo_kmeans

def avaliar_modelo(X_scaled, modelo_kmeans):
    """
    Avalia a qualidade do modelo usando Silhouette Score e Inércia.
    """
    score = silhouette_score(X_scaled, modelo_kmeans.labels_)
    inercia = modelo_kmeans.inertia_
    
    print("--- AVALIAÇÃO DO MODELO K-MEANS ---")
    print(f"Silhouette Score: {score:.4f}")
    print(f"Inércia: {inercia:.2f}\n")
    return score