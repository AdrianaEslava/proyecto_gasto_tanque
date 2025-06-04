import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import joblib
import numpy as np
import sys
import os

# Agregar la ruta correcta para la importación del modelo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modelo import GastoNN

def entrenar_modelo(ruta_datos, ruta_guardado_modelo):
    # Cargar datos
    df = pd.read_csv(ruta_datos, delimiter=';', decimal=',')
    X = df[['frecuencia_inicial', 'amplitud_inicial', 'frecuencia_final', 'amplitud_final']].values
    y = df[['gasto']].values

    # Filtrar valores NaN en y antes del escalado
    y = y[~np.isnan(y)]
    if y.size == 0:
        raise ValueError("Error: No hay valores válidos en 'gasto'. Revisa el archivo CSV.")

    # Asegurar que y tenga el formato correcto antes de escalar
    y = y.reshape(-1, 1)
    
    print("Valores originales de y:", y[:5])
    print("Mínimo y Máximo de y antes del escalado:", y.min(), y.max())

    # Ajustar valores pequeños para evitar NaN
    y += 1e-6  # Pequeño desplazamiento para evitar problemas con valores cero
    y *= 1000  # Escalamos los valores para que no sean demasiado pequeños

    # Aplicamos logaritmo si los valores son positivos
    y = np.log1p(y) if np.min(y) > 0 else y

    # Escalado con RobustScaler para mejorar estabilidad
    scaler_y = RobustScaler()
    y_scaled = scaler_y.fit_transform(y)

    print("Valores de y después de ajuste:", y[:5])
    print("Valores de y_scaled después del escalado:", y_scaled[:5])

    # Verificar que no haya valores NaN en y_scaled
    if np.isnan(y_scaled).any():
        raise ValueError("Error crítico: Se encontraron NaN en y_scaled después del escalado. Revisa los datos antes de continuar.")

    # Verificar si X contiene NaN antes del escalado
    print("Valores de X antes del escalado:", X[:5])
    print("Hay NaN en X antes del escalado?", np.isnan(X).any())

    # Filtrar y corregir valores NaN en X antes del escalado
    X = np.nan_to_num(X, nan=np.nanmean(X))  # Sustituye NaN por el promedio de la columna
    X = X[~np.isnan(X).any(axis=1)]  # Elimina filas con valores NaN

    # Escalado de datos para X
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Verificar que no haya valores NaN en X_scaled
    if np.isnan(X_scaled).any():
        raise ValueError("Error crítico: Se encontraron NaN en X_scaled después de filtrar. Revisa los datos antes de continuar.")

    # Asegurar que X_scaled y y_scaled tengan el mismo número de filas
    min_samples = min(X_scaled.shape[0], y_scaled.shape[0])  # Obtener el tamaño mínimo
    X_scaled = X_scaled[:min_samples]  # Ajustar X_scaled
    y_scaled = y_scaled[:min_samples]  # Ajustar y_scaled

    print("Forma final de X_scaled:", X_scaled.shape)
    print("Forma final de y_scaled:", y_scaled.shape)

    # Convertir datos escalados a tensores
    inputs = torch.tensor(X_scaled, dtype=torch.float32)
    targets = torch.tensor(y_scaled, dtype=torch.float32)

    print("Hay NaN en inputs:", torch.isnan(inputs).any())
    print("Hay NaN en targets:", torch.isnan(targets).any())

    # Definir modelo
    input_size = inputs.shape[1]
    model = GastoNN(input_size)

    # Definir función de pérdida y optimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Entrenamiento
    epochs = 500
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}')

    # Guardar modelo y escaladores
    torch.save(model.state_dict(), f"{ruta_guardado_modelo}/modelo_gasto.pth")
    joblib.dump(scaler_X, f"{ruta_guardado_modelo}/scaler_X.pkl")
    joblib.dump(scaler_y, f"{ruta_guardado_modelo}/scaler_y.pkl")
    print("Modelo y escaladores guardados exitosamente.")
