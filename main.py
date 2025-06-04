from src.entrenamiento import entrenar_modelo
from src.prediccion import predecir_gasto
import sys
import os

def main():
    
# Agrega el directorio raíz del proyecto al sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

    # Rutas de archivos
    ruta_datos_entrenamiento = 'datos/datos_entrenamiento.csv'
    ruta_modelo = 'modelos'
    ruta_datos_nuevos = 'datos/datos_nuevos.csv'
    ruta_resultado = 'datos/predicciones_gasto.csv'

    # Entrenar el modelo
    # entrenar_modelo(ruta_datos_entrenamiento, ruta_modelo)

    # Realizar predicciones
    predecir_gasto(ruta_datos_nuevos, ruta_modelo, ruta_resultado)

if __name__ == "__main__":
    main()
