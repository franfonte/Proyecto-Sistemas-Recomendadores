import pandas as pd
import numpy as np
import os
import sys

def preprocess_data(data_path):
    """
    Carga los datos de entrenamiento y el conjunto antitest para el modelo aleatorio.
    """
    print(f"1. Preprocesando datos desde: {data_path}")
    train_file = os.path.join(data_path, 'train.csv')
    antitest_file = os.path.join(data_path, 'antitest.csv')
    
    train_df = pd.read_csv(train_file)
    antitest_df = pd.read_csv(antitest_file)
    
    print("   Datos cargados en DataFrames.")
    return train_df, antitest_df

def train_model(train_df):
    """
    El modelo aleatorio no requiere entrenamiento.
    Esta función es un marcador de posición.
    """
    print("2. El modelo aleatorio no necesita entrenamiento.")
    # Se devuelve un diccionario con el rango de posibles valoraciones
    model_info = {'min_rating': 1.0, 'max_rating': 5.0}
    return model_info

def generate_predictions(model_info, antitest_df):
    """
    Genera predicciones aleatorias para cada par (usuario, item) en el antitest set.
    """
    print("3. Generando predicciones aleatorias...")
    
    min_rating = model_info['min_rating']
    max_rating = model_info['max_rating']
    
    # Genera un número aleatorio para cada fila en el antitest_df
    predictions = np.random.uniform(min_rating, max_rating, len(antitest_df))
    
    # Crea una copia para evitar SettingWithCopyWarning
    predictions_df = antitest_df.copy()
    predictions_df['prediction'] = predictions
    
    print(f"   Se generaron {len(predictions_df)} predicciones.")
    return predictions_df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Por favor, proporciona el porcentaje del dataset (ej. 10, 25, ...).")
        sys.exit(1)
    
    dataset_percentage = sys.argv[1]
    DATA_PATH = os.path.join('data', dataset_percentage)
    
    # 1. Preprocesar datos
    train_data, antitest_data = preprocess_data(DATA_PATH)
    
    # 2. "Entrenar" el modelo
    model = train_model(train_data)
    
    # 3. Generar predicciones
    predictions = generate_predictions(model, antitest_data)
    
    print("\n--- Proceso del modelo aleatorio finalizado ---")
    print("Ejemplo de 5 predicciones:")
    print(predictions.head())
