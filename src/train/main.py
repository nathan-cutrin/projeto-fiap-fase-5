# main.py
from train.data_processing import carregar_dados, renomear_colunas, preparar_dados, padronizar_dados
from train.model_training import treinar_modelo
from train.utils import salvar_modelo
from train.config import modelo_path, scaler_path

# Carregar e preparar os dados
df_2022, df_2023, df_2024 = carregar_dados()
df_2022, df_2023, df_2024 = renomear_colunas(df_2022, df_2023, df_2024)
df_2024 = preparar_dados(df_2022, df_2023, df_2024)

# Padronizar os dados
X_scaled, scaler = padronizar_dados(df_2024)

# Treinar o modelo
modelo = treinar_modelo(X_scaled)

# Salvar o modelo e o scaler
salvar_modelo(modelo, scaler, modelo_path, scaler_path)