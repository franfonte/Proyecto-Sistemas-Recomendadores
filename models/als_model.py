import pandas as pd
import numpy as np
import os
import sys
from scipy.sparse import csr_matrix
import implicit # Nueva librería para ALS

# --- Hiperparámetros por Defecto ---
FACTORS = 64
EPOCHS = 15
REGULARIZATION = 0.01

def preprocess_data(data_path):
    """
    Carga datos y los convierte al formato que 'implicit' ALS necesita:
    - Binariza los ratings: >= 4.0 es 1 (positivo), < 4.0 se ignora.
    - Crea mapeos de ID GLOBALES basados en train + antitest.
    - Crea una matriz de interacciones dispersa (CSR) con dimensiones GLOBALES,
      rellenada solo con las interacciones positivas.
    """
    print(f"1. Preprocesando datos para Implicit ALS desde: {data_path}")
    train_file = os.path.join(data_path, 'train.csv')
    antitest_file = os.path.join(data_path, 'antitest.csv')

    train_df = pd.read_csv(train_file)
    antitest_df = pd.read_csv(antitest_file)

    # <<<<<<< CAMBIO CRÍTICO: Mapas GLOBALES >>>>>>>
    # Crear mapeos de IDs a índices enteros (basados en TODOS los usuarios/items posibles)
    all_users = pd.concat([train_df['userId'], antitest_df['userId']]).unique()
    all_items = pd.concat([train_df['movieId'], antitest_df['movieId']]).unique()
    user_map = {uid: i for i, uid in enumerate(all_users)}
    item_map = {iid: i for i, iid in enumerate(all_items)}
    num_users = len(user_map) # Dimensiones globales
    num_items = len(item_map) # Dimensiones globales
    print(f"   Usuarios únicos totales (train+antitest): {num_users}")
    print(f"   Items únicos totales (train+antitest): {num_items}")

    # --- Lógica de Binarización (Tu regla) ---
    print("   Aplicando regla de binarización: rating >= 4.0 --> 1, < 4.0 se ignora")
    train_df.loc[:, 'rating_bin'] = train_df['rating'].apply(lambda x: 1 if x >= 4.0 else 0)
    train_positive = train_df[train_df['rating_bin'] == 1].copy()

    # Mapear IDs a índices GLOBALES en el DataFrame de entrenamiento positivo
    train_positive.loc[:, 'user_idx'] = train_positive['userId'].map(user_map)
    train_positive.loc[:, 'item_idx'] = train_positive['movieId'].map(item_map)
    # Eliminar filas si algún ID positivo no estuviera en los mapas globales (poco probable, pero seguro)
    train_positive = train_positive.dropna(subset=['user_idx', 'item_idx'])
    train_positive.loc[:, 'user_idx'] = train_positive['user_idx'].astype(int)
    train_positive.loc[:, 'item_idx'] = train_positive['item_idx'].astype(int)

    print(f"   Interacciones positivas (>=4) a usar para entrenamiento: {len(train_positive)}")

    # Crear la matriz de interacciones dispersa (CSR format) con las dimensiones GLOBALES
    interactions_matrix = csr_matrix(
        (np.ones(len(train_positive)), (train_positive['user_idx'], train_positive['item_idx'])), # Usar 1s como valor
        shape=(num_users, num_items) # Usar las dimensiones GLOBALES
    )
    print("   Datos cargados y convertidos a matriz dispersa CSR.")

    # El modelo ALS se entrena con (items, users)
    training_data = interactions_matrix.T.tocsr()

    prediction_components = {
        'antitest_df': antitest_df,
        'user_map': user_map, # Pasar los mapas GLOBALES
        'item_map': item_map  # Pasar los mapas GLOBALES
    }
    return training_data, prediction_components

def train_model(interactions_matrix_transposed):
    """
    Entrena un modelo ALS de la librería 'implicit'.
    """
    print("2. Entrenando el modelo Implicit ALS...")
    model = implicit.als.AlternatingLeastSquares(
        factors=FACTORS,
        regularization=REGULARIZATION,
        iterations=EPOCHS,
        random_state=42
    )
    model.fit(interactions_matrix_transposed)
    print("   Entrenamiento completado.")
    return model

def generate_predictions(model, prediction_components):
    """
    Genera predicciones para el conjunto antitest con ALS.
    Calcula el producto punto de los factores de usuario e ítem.
    """
    print("3. Generando predicciones con Implicit ALS...")

    antitest_df = prediction_components['antitest_df']
    user_map = prediction_components['user_map'] # Usar mapa GLOBAL
    item_map = prediction_components['item_map'] # Usar mapa GLOBAL

    # Obtener los factores finales del modelo
    user_factors_from_model = model.user_factors
    item_factors_from_model = model.item_factors
    num_users_model = user_factors_from_model.shape[0]
    num_items_model = item_factors_from_model.shape[0]
    print(f"   Dimensiones correctas según el modelo: {num_users_model} usuarios, {num_items_model} items.")

    # Mapear IDs del antitest usando mapas GLOBALES
    antitest_mapped = antitest_df.copy()
    antitest_mapped['user_idx'] = antitest_mapped['userId'].map(user_map)
    antitest_mapped['item_idx'] = antitest_mapped['movieId'].map(item_map)

    # Filtrar solo si el mapeo falló
    valid_antitest_mapped = antitest_mapped.dropna(subset=['user_idx', 'item_idx']).copy()
    valid_antitest_mapped.loc[:, 'user_idx'] = valid_antitest_mapped['user_idx'].astype(int)
    valid_antitest_mapped.loc[:, 'item_idx'] = valid_antitest_mapped['item_idx'].astype(int)

    # Obtener los índices directamente
    user_indices_all = valid_antitest_mapped['user_idx'].values.astype(np.int64)
    item_indices_all = valid_antitest_mapped['item_idx'].values.astype(np.int64)

    # <<<<<<< CAMBIO CRÍTICO: Filtro explícito ANTES de indexar >>>>>>>
    # Crear máscaras para asegurar que los índices estén DENTRO de los límites
    user_mask = user_indices_all < num_users_model
    item_mask = item_indices_all < num_items_model
    valid_mask = user_mask & item_mask

    # Aplicar la máscara A LOS ÍNDICES y al DataFrame CORRESPONDIENTE
    user_indices_valid = user_indices_all[valid_mask]
    item_indices_valid = item_indices_all[valid_mask]
    predictions_df = valid_antitest_mapped[valid_mask].copy() # Usar la máscara en el DataFrame también

    if len(predictions_df) < len(valid_antitest_mapped):
         print(f"   Advertencia: Se descartaron {len(valid_antitest_mapped) - len(predictions_df)} filas con índices fuera de rango ANTES de la predicción.")

    # Calcular el producto punto SOLAMENTE con los índices válidos
    predictions = (user_factors_from_model[user_indices_valid] * item_factors_from_model[item_indices_valid]).sum(axis=1)

    # Asignar las predicciones al DataFrame YA FILTRADO
    predictions_df['prediction'] = predictions

    print(f"   Se generaron {len(predictions_df)} predicciones.")
    # Devolver las columnas originales + la predicción
    return predictions_df[['userId', 'movieId', 'prediction']]