import joblib
from pathlib import Path

ml_models = {}

def load_ml_artifacts():
    """
    Carrega o modelo K-Means e o Scaler treinados na memória usando caminhos absolutos.
    """
    # Descobre automaticamente onde este arquivo (model_services.py) está no computador
    caminho_atual = Path(__file__).resolve()
    
    # Navega para trás até chegar na pasta 'src'
    # caminho_atual = src/api/services/model_services.py
    # .parent = services -> .parent = api -> .parent = src
    pasta_src = caminho_atual.parent.parent.parent
    
    # Monta o caminho exato para a pasta models
    model_path = pasta_src / "models" / "model.pkl"
    scaler_path = pasta_src / "models" / "scaler.pkl"
    
    print(f"🔎 Procurando modelo em: {model_path}")
    print(f"🔎 Procurando scaler em: {scaler_path}")
    
    try:
        # Carregando o modelo
        if model_path.exists():
            ml_models["passos_magicos_model"] = joblib.load(model_path)
            print("✅ Modelo carregado com sucesso!")
        else:
            print(f"❌ AVISO: model.pkl não encontrado!")

        # Carregando o scaler
        if scaler_path.exists():
            ml_models["passos_magicos_scaler"] = joblib.load(scaler_path)
            print("✅ Scaler carregado com sucesso!")
        else:
            print(f"❌ AVISO: scaler.pkl não encontrado!")
            
    except Exception as e:
        print(f"❌ Erro fatal ao carregar artefatos: {e}")