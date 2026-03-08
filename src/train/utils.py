# src/model_utils.py
import joblib

def salvar_modelo(modelo_hc, scaler, modelo_path='model.pkl', scaler_path='scaler.pkl'):
    """
    Função para salvar o modelo treinado e o scaler usando joblib.
    """
    joblib.dump(modelo_hc, modelo_path)
    joblib.dump(scaler, scaler_path)
    print(f"Modelo e scaler salvos em: {modelo_path}, {scaler_path}")

def carregar_modelo(modelo_path='model.pkl', scaler_path='scaler.pkl'):
    """
    Função para carregar o modelo treinado e o scaler com joblib.
    """
    modelo_hc = joblib.load(modelo_path)
    scaler = joblib.load(scaler_path)
    print(f"Modelo e scaler carregados de: {modelo_path}, {scaler_path}")
    
    return modelo_hc, scaler