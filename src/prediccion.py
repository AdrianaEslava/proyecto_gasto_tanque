import pandas as pd
import torch
import joblib
from src.modelo import GastoNN

def predecir_gasto(ruta_datos_nuevos, ruta_modelo, ruta_resultado):
    # Cargar escaladores
    scaler_X = joblib.load(f"{ruta_modelo}/scaler_X.pkl")
    scaler_y = joblib.load(f"{ruta_modelo}/scaler_y.pkl")

    # Cargar modelo
    input_size = 4  # Número de características
    model = GastoNN(input_size)
    model.load_state_dict(torch.load(f"{ruta_modelo}/modelo_gasto.pth"))
    model.eval()

    # Cargar nuevos datos
    df_nuevos = pd.read_csv(ruta_datos_nuevos, delimiter=';', decimal=',', header=0)
    df_nuevos = df_nuevos.dropna(axis=1, how="all")  # Elimina columnas vacías
    print("Columnas del archivo CSV:", df_nuevos.columns)

    print(df_nuevos.head())
    df_nuevos.columns = df_nuevos.columns.str.strip()

    X_nuevos = df_nuevos[['amplitud_inicial','frecuencia_inicial', 'frecuencia_final', 'amplitud_final']].values
    X_nuevos_scaled = scaler_X.transform(X_nuevos)
    inputs_nuevos = torch.tensor(X_nuevos_scaled, dtype=torch.float32)

    # Realizar predicciones
    with torch.no_grad():
        predicciones_scaled = model(inputs_nuevos)
        predicciones = scaler_y.inverse_transform(predicciones_scaled.numpy())

    # Guardar resultados
    df_nuevos['gasto_estimado'] = predicciones
    df_nuevos.to_csv(ruta_resultado, index=False)
    print(f"Predicciones guardadas en '{ruta_resultado}'.")
