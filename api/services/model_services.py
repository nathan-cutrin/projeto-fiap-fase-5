import joblib
import pandas as pd
from pathlib import Path

ml_models = {
    "passos_magicos_model": None,
    "passos_magicos_scaler": None,
    "student_database": None
}

def load_ml_artifacts():
    workdir = Path(__file__).resolve()
    pasta_api = workdir.parent.parent 
    
    model_path = pasta_api.parent / "models" / "model.pkl"
    scaler_path = pasta_api.parent / "models" / "scaler.pkl"
    db_path = pasta_api / "database" / "alunos_db.csv"

    try:
        if model_path.exists():
            ml_models["passos_magicos_model"] = joblib.load(model_path)
        if scaler_path.exists():
            ml_models["passos_magicos_scaler"] = joblib.load(scaler_path)

        if db_path.exists():
            df_temp = pd.read_csv(db_path)
            df_limpo = df_temp.where(pd.notnull(df_temp), None)
            
            df_limpo['ra'] = df_limpo['ra'].astype(str).str.strip()
            
            ml_models["student_database"] = df_limpo
            print(f"✅ Banco de dados carregado: {len(df_limpo)} alunos prontos.")
            
    except Exception as e:
        print(f"❌ Erro ao carregar artefatos: {e}")