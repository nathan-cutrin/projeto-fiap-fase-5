import joblib
import os

# Dicionário que vai guardar nossos artefatos em memória
ml_models = {}

def load_ml_artifacts():
    """
    Carrega o modelo K-Means e o Scaler treinados na memória.
    Os caminhos assumem que você roda a API a partir da raiz do projeto (projeto-5).
    """
    model_path = "src/models/model.pkl"
    scaler_path = "src/models/scaler.pkl"
    
    try:
        # Carregando o modelo
        if os.path.exists(model_path):
            ml_models["passos_magicos_model"] = joblib.load(model_path)
        else:
            print(f"Aviso: Modelo não encontrado em {model_path}")

        # Carregando o scaler
        if os.path.exists(scaler_path):
            ml_models["passos_magicos_scaler"] = joblib.load(scaler_path)
        else:
            print(f"Aviso: Scaler não encontrado em {scaler_path}")
            
        print("Artefatos de Machine Learning carregados com sucesso!")
        
    except Exception as e:
        print(f"Erro fatal ao carregar artefatos: {e}")