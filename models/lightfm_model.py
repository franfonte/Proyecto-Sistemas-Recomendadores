import pandas as pd
import numpy as np
import os
import sys
from scipy.sparse import coo_matrix
from lightfm import LightFM

def preprocess_data(data_path):
    """
    Carga datos y los convierte al formato que LightFM necesita:
    una matriz de interacciones dispersa y mapeos de ID.
    """
    print(f"1. Preprocesando datos para LightFM desde: {data_path}")
    train_file = os.path.join(data_path, 'train.csv')
    antitest_file = os.path.join(data_path, 'antitest.csv')

    train_df = pd.read_csv(train_file)
    antitest_df = pd.read_csv(antitest_file)

    # Crear mapeos de IDs a índices enteros
    user_id_map = {id: i for i, id in enumerate(train_df['userId'].unique())}
    item_id_map = {id: i for i, id in enumerate(train_df['movieId'].unique())}
    
    # Mapear IDs en los dataframes
    train_df['user_idx'] = train_df['userId'].map(user_id_map)
    train_df['item_idx'] = train_df['movieId'].map(item_id_map)

    # Crear la matriz de interacciones dispersa (COO format)
    interactions = coo_matrix((train_df['rating'], (train_df['user_idx'], train_df['item_idx'])))

    print("   Datos cargados y convertidos a matriz dispersa.")
    return interactions, antitest_df, user_id_map, item_id_map

def train_model(interactions):
    """
    Entrena un modelo LightFM.
    """
    print("2. Entrenando el modelo LightFM...")
    # Usaremos el algoritmo 'warp' (Weighted Approximate-Rank Pairwise)
    model = LightFM(loss='warp', random_state=42)
    model.fit(interactions, epochs=10, num_threads=4)
    print("   Entrenamiento completado.")
    return model

def generate_predictions(model, antitest_df, user_id_map, item_id_map):
    """
    Genera predicciones para el conjunto antitest con LightFM.
    """
    print("3. Generando predicciones con LightFM...")
    
    # Mapear IDs del antitest. Ignorar usuarios/items no vistos en el entrenamiento.
    antitest_df['user_idx'] = antitest_df['userId'].map(user_id_map)
    antitest_df['item_idx'] = antitest_df['movieId'].map(item_id_map)
    
    # Filtrar filas donde el usuario o item no estaba en el set de entrenamiento
    valid_antitest = antitest_df.dropna(subset=['user_idx', 'item_idx'])
    valid_antitest['user_idx'] = valid_antitest['user_idx'].astype(int)
    valid_antitest['item_idx'] = valid_antitest['item_idx'].astype(int)

    # Generar predicciones (scores)
    scores = model.predict(
        valid_antitest['user_idx'].values,
        valid_antitest['item_idx'].values,
        num_threads=4
    )
    
    predictions_df = valid_antitest.copy()
    predictions_df['prediction'] = scores
    
    print(f"   Se generaron {len(predictions_df)} predicciones.")
    return predictions_df[['userId', 'movieId', 'prediction']]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Por favor, proporciona el porcentaje del dataset (ej. 10, 25, ...).")
        sys.exit(1)
    
    dataset_percentage = sys.argv[1]
    DATA_PATH = os.path.join('data', dataset_percentage)

    # 1. Preprocesar datos
    interactions_matrix, antitest_data, user_map, item_map = preprocess_data(DATA_PATH)
    
    # 2. Entrenar el modelo
    trained_model = train_model(interactions_matrix)
    
    # 3. Generar predicciones
    predictions = generate_predictions(trained_model, antitest_data, user_map, item_map)
    
    print("\n--- Proceso del modelo LightFM finalizado ---")
    print("Ejemplo de 5 predicciones:")
    print(predictions.head())
