import pandas as pd
import numpy as np
import os
import sys
from scipy.sparse import coo_matrix
from lightfm import LightFM
import random # Necesario para random.seed (aunque LightFM lo maneja internamente)

# --- Constante para Replicabilidad (usada en train_model) ---
RANDOM_SEED = 42

def preprocess_data(data_path):
    """
    Carga datos y los convierte al formato que LightFM necesita:
    una matriz de interacciones dispersa y mapeos de ID basados en train.
    AHORA DEVUELVE LOS MAPAS en el formato esperado por el runner.
    """
    print(f"1. Preprocesando datos para LightFM desde: {data_path}")
    train_file = os.path.join(data_path, 'train.csv')
    antitest_file = os.path.join(data_path, 'antitest.csv')

    train_df = pd.read_csv(train_file)
    antitest_df = pd.read_csv(antitest_file)

    # Crear mapeos de IDs a índices enteros (basados solo en train)
    user_ids_train = train_df['userId'].unique()
    item_ids_train = train_df['movieId'].unique()
    user_id_map = {id: i for i, id in enumerate(user_ids_train)}
    item_id_map = {id: i for i, id in enumerate(item_ids_train)}
    num_users_train = len(user_id_map)
    num_items_train = len(item_id_map)
    print(f"   Usuarios únicos en train: {num_users_train}")
    print(f"   Items únicos en train: {num_items_train}")

    # Mapear IDs en los dataframes de train
    train_df['user_idx'] = train_df['userId'].map(user_id_map)
    train_df['item_idx'] = train_df['movieId'].map(item_id_map)
    # Eliminar filas si algún ID no estuviera en los mapas (no debería pasar)
    train_df = train_df.dropna(subset=['user_idx', 'item_idx'])
    train_df['user_idx'] = train_df['user_idx'].astype(int)
    train_df['item_idx'] = train_df['item_idx'].astype(int)

    # Crear la matriz de interacciones dispersa (COO format) con dimensiones de train
    # Usar float32 para consistencia con otros modelos
    interactions = coo_matrix(
        (train_df['rating'].astype(np.float32), (train_df['user_idx'], train_df['item_idx'])),
        shape=(num_users_train, num_items_train)
    )

    print("   Datos cargados y convertidos a matriz dispersa.")

    # <<<<<<< CAMBIO AQUÍ: Reorganizar el retorno >>>>>>>
    training_components = interactions # Datos para entrenar
    prediction_components = { # Argumentos extra para predecir
        'antitest_df': antitest_df
        # Los mapas se pasarán por separado ahora
    }
    # Devolver training_components, prediction_components, user_map, item_map
    return training_components, prediction_components, user_id_map, item_id_map

def train_model(interactions):
    """
    Entrena un modelo LightFM.
    """
    # Fijar semillas para replicabilidad
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    print(f"   Semillas NumPy y Python fijadas en: {RANDOM_SEED}")

    print("2. Entrenando el modelo LightFM...")
    model = LightFM(loss='warp', random_state=RANDOM_SEED)
    # Usar 1 hilo para asegurar determinismo
    model.fit(interactions, epochs=10, num_threads=1)
    print("   Entrenamiento completado.")
    return model

# <<<<<<< CAMBIO AQUÍ: Firma de la función >>>>>>>
def generate_predictions(model, prediction_components):
    """
    Genera predicciones para el conjunto antitest con LightFM.
    Ahora recibe los componentes en un diccionario.
    """
    print("3. Generando predicciones con LightFM...")

    # <<<<<<< CAMBIO AQUÍ: Desempaquetar componentes >>>>>>>
    antitest_df = prediction_components['antitest_df']
    # Los mapas ahora vienen del runner a través del diccionario
    user_id_map = prediction_components['user_map']
    item_id_map = prediction_components['item_map']

    # --- El resto de la lógica permanece igual ---

    # Mapear IDs del antitest. Ignorar usuarios/items no vistos en el entrenamiento.
    antitest_mapped = antitest_df.copy()
    antitest_mapped['user_idx'] = antitest_mapped['userId'].map(user_id_map)
    antitest_mapped['item_idx'] = antitest_mapped['movieId'].map(item_id_map)

    # Filtrar filas donde el usuario o item no estaba en el set de entrenamiento
    valid_antitest = antitest_mapped.dropna(subset=['user_idx', 'item_idx']).copy()
    valid_antitest['user_idx'] = valid_antitest['user_idx'].astype(int)
    valid_antitest['item_idx'] = valid_antitest['item_idx'].astype(int)

    # Obtener dimensiones del modelo a partir de los mapas
    num_users_model = len(user_id_map)
    num_items_model = len(item_id_map)

    # Filtrado adicional por si acaso
    valid_antitest = valid_antitest[
        (valid_antitest['user_idx'] < num_users_model) &
        (valid_antitest['item_idx'] < num_items_model)
    ].copy()

    # Convertir a int32 explícitamente como espera LightFM
    user_indices = valid_antitest['user_idx'].values.astype(np.int32)
    item_indices = valid_antitest['item_idx'].values.astype(np.int32)

    if len(user_indices) == 0 or len(item_indices) == 0:
         print("   Advertencia: No hay pares usuario-ítem válidos para predecir después del filtrado.")
         return pd.DataFrame(columns=['userId', 'movieId', 'prediction'])

    # Generar predicciones (scores) con 1 hilo
    scores = model.predict(
        user_indices,
        item_indices,
        num_threads=1
    )

    predictions_df = valid_antitest.copy()
    predictions_df['prediction'] = scores

    print(f"   Se generaron {len(predictions_df)} predicciones.")
    return predictions_df[['userId', 'movieId', 'prediction']]

# Bloque if __name__ == "__main__" (ajustado para reflejar cambios)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python lightfm_model.py <dataset_percentage>")
        sys.exit(1)

    dataset_percentage = sys.argv[1]
    # Asume que se corre desde el directorio raíz del proyecto
    DATA_PATH = os.path.join('data', dataset_percentage)

    if not os.path.exists(DATA_PATH):
        print(f"Error: El directorio de datos no existe en '{DATA_PATH}'")
        sys.exit(1)

    try:
        # 1. Preprocesar datos
        interactions_matrix, pred_comps_dict, u_map, i_map = preprocess_data(DATA_PATH)
        # Añadir mapas al diccionario para generate_predictions si se corre individualmente
        pred_comps_dict['user_map'] = u_map
        pred_comps_dict['item_map'] = i_map

        # 2. Entrenar el modelo
        trained_model = train_model(interactions_matrix)

        # 3. Generar predicciones
        predictions = generate_predictions(trained_model, pred_comps_dict)

        print("\n--- Proceso del modelo LightFM finalizado ---")
        if predictions is not None and not predictions.empty:
             print("Ejemplo de 5 predicciones:")
             print(predictions.head())
        else:
             print("No se generaron predicciones.")

    except FileNotFoundError:
        print(f"Error: No se encontraron los archivos train.csv o antitest.csv en '{DATA_PATH}'")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        import traceback
        traceback.print_exc()
