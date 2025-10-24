import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import pickle
import random
import time
from codecarbon import EmissionsTracker

# Importaciones de librerías de modelos
import torch
from surprise.dump import load as surprise_load
from lightfm import LightFM
import implicit

# Importar Clases de Modelos PyTorch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))
try:
    from ncf_model import NCF
    from lightgcn_model import LightGCN
    from multivae_model import MultiVAE
    # Importar las funciones de preproceso para reconstruir dependencias
    from lightgcn_model import create_norm_adj_matrix as lightgcn_create_matrix
    from multivae_model import VAEInteractionDataset # Necesario para predicción
except ImportError as e:
    print(f"Error importando clases de modelos PyTorch: {e}")
    # Definir placeholders para que el script no falle si faltan archivos
    class NCF: pass
    class LightGCN: pass
    class MultiVAE: pass

# --- Constante para Replicabilidad ---
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(RANDOM_SEED)

# --- Funciones de Carga (Similares a predict_single.py) ---

def load_maps_from_json(model_dir, model_name):
    """Carga user_map y item_map desde archivos JSON."""
    user_map_path = os.path.join(model_dir, f"{model_name}_user_map.json")
    item_map_path = os.path.join(model_dir, f"{model_name}_item_map.json")
    user_map, item_map = None, None
    try:
        with open(user_map_path, 'r') as f:
            user_map_str_keys = json.load(f)
            user_map = {int(k): v for k, v in user_map_str_keys.items()}
        with open(item_map_path, 'r') as f:
            item_map_str_keys = json.load(f)
            item_map = {int(k): v for k, v in item_map_str_keys.items()}
        print("   Mapas de ID cargados desde JSON.")
    except Exception as e:
        print(f"   [ERROR] Falló al cargar mapas JSON para {model_name}: {e}")
    return user_map, item_map

def load_pytorch_model(model_class, state_dict_path, user_map, item_map, **kwargs):
    """Carga un modelo PyTorch."""
    num_users = len(user_map)
    num_items = len(item_map)
    model_instance = None
    try:
        if model_class == MultiVAE:
             model_instance = model_class(num_items=num_items, hidden_dim=600, latent_dim=200) # Hiperparams fijos
        elif model_class == NCF:
             model_instance = model_class(num_users=num_users, num_items=num_items, embedding_dim=64, mlp_layers=[64, 32, 16]) # Hiperparams fijos
        elif model_class == LightGCN:
             # LightGCN necesita la matriz de adyacencia
             norm_adj_matrix = kwargs.get('norm_adj_matrix')
             if norm_adj_matrix is None:
                 print("[ERROR] LightGCN necesita 'norm_adj_matrix' para ser instanciado.")
                 return None
             model_instance = model_class(num_users=num_users, num_items=num_items, num_layers=3, embedding_dim=64, norm_adj_matrix=norm_adj_matrix)
        
        if model_instance:
            model_instance.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
            model_instance.eval()
            print(f"   Modelo {model_class.__name__} cargado correctamente.")
            return model_instance
    except Exception as e:
        print(f"[ERROR] Falló al instanciar o cargar state_dict de {model_class.__name__}: {e}")
    return None

# (Otras funciones de carga de predict_single.py: load_surprise_model, load_pickle_model, load_most_popular)
def load_surprise_model(model_path):
    try:
        _, model = surprise_load(model_path)
        print("   Modelo Surprise cargado.")
        return model
    except Exception as e:
        print(f"[ERROR] Falló al cargar modelo Surprise: {e}")
        return None

def load_pickle_model(model_path, model_name):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"   Modelo {model_name} (pickle) cargado.")
        return model
    except Exception as e:
        print(f"[ERROR] Falló al cargar modelo Pickle ({model_name}): {e}")
        return None

def load_most_popular(model_path):
    try:
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        popularity_scores = {int(k): v for k, v in model_data['popularity_scores'].items()}
        global_average = model_data['global_average']
        print("   Modelo Most Popular cargado.")
        return pd.Series(popularity_scores), global_average
    except Exception as e:
        print(f"[ERROR] Falló al cargar modelo Most Popular: {e}")
        return None, None

# --- Funciones de Predicción Individual (Adaptadas) ---

# Reutilizar generate_predictions de los módulos de modelo es la mejor estrategia
# Esta función adaptará la llamada
def get_model_predictions_for_user(model_module, model, prediction_components_user):
    """
    Wrapper para llamar a la función generate_predictions de cada modelo
    con los datos filtrados para un solo usuario.
    """
    # model_module es el módulo importado (ej. ncf_model)
    # model es el objeto de modelo entrenado y cargado
    # prediction_components_user es el dict con 'antitest_df' (filtrado), 'user_map', 'item_map', etc.
    
    # La firma de generate_predictions es: (model, prediction_components)
    return model_module.generate_predictions(model, prediction_components_user)

# --- Función de Guardado de Resultados ---

def save_individual_results(filepath, dataset_percentage, model_name, user_id, top_10_movie_ids, footprint):
    """
    Carga y actualiza el JSON de resultados individuales.
    """
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = {}
    else:
        all_results = {}

    dataset_key = str(dataset_percentage)
    user_key = str(user_id) # Usar string para claves JSON

    # Crear estructura anidada
    if dataset_key not in all_results:
        all_results[dataset_key] = {}
    if model_name not in all_results[dataset_key]:
        all_results[dataset_key][model_name] = {}
    
    # Datos a guardar
    results_data = {
        "top_10_recommendations": top_10_movie_ids,
        "prediction_footprint": {
            "co2_emissions_g": footprint * 1000 if isinstance(footprint, (int, float)) else None,
            # (Podríamos añadir duration y energy si las extraemos del tracker)
        }
    }
    
    all_results[dataset_key][model_name][user_key] = results_data

    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=4, sort_keys=True)
    
    print(f"\nResultados individuales guardados en '{filepath}' para el usuario {user_id}.")


# --- Función Principal ---
def main(args):
    print(f"--- Generando Top-10 para Usuario: {args.user_id} ---")
    print(f"Modelo: {args.model_name}, Dataset: {args.dataset_percentage}%")

    # --- 1. Definir Rutas ---
    data_path = os.path.join('data', args.dataset_percentage)
    model_dir = os.path.join('trained_models', args.dataset_percentage)
    results_file = 'individual_results.json'

    # Rutas del modelo y mapas
    extension = '.pkl'
    is_pytorch_model = args.model_name in ['ncf_model', 'lightgcn_model', 'multivae_model']
    if is_pytorch_model: extension = '.pth'
    elif args.model_name == 'most_popular_model': extension = '.json'
    model_path = os.path.join(model_dir, f"{args.model_name}{extension}")

    # --- 2. Cargar Módulo de Modelo ---
    try:
        model_module = importlib.import_module(f"models.{args.model_name}")
    except ImportError:
        print(f"[ERROR] No se pudo encontrar el módulo 'models/{args.model_name}.py'.")
        return

    # --- 3. Cargar Mapas (si es necesario) ---
    user_map, item_map = None, None
    needs_maps = args.model_name not in ['svd_model', 'item_knn_model', 'user_knn_model', 'most_popular_model', 'random_model']
    if needs_maps:
        user_map, item_map = load_maps_from_json(model_dir, args.model_name)
        if user_map is None or item_map is None:
            print("[ERROR] No se pudieron cargar los mapas necesarios para este modelo.")
            return

    # --- 4. Cargar el Modelo Entrenado ---
    model = None
    kwargs = {} # Argumentos extra para constructores de PyTorch
    
    # Lógica especial para modelos que necesitan dependencias reconstruidas
    if args.model_name in ['lightgcn_model', 'multivae_model', 'als_model']:
        # Reconstruir las dependencias que estos modelos necesitan (mapas, matrices)
        print("   Reconstruyendo dependencias de preproceso (mapas, matrices)...")
        # (Esto es una simplificación, llamamos a preprocess_data solo para obtener los mapas
        #  y la matriz de LightGCN. Idealmente, guardaríamos la matriz también.)
        try:
             # Llamar a la función preprocess del módulo específico
             temp_train, temp_pred, temp_u_map, temp_i_map = model_module.preprocess_data(data_path)
             if args.model_name == 'lightgcn_model':
                 kwargs['norm_adj_matrix'] = temp_train['norm_adj_matrix']
             elif args.model_name == 'multivae_model':
                 # MultiVAE necesita la matriz de interacciones para predicción
                 kwargs['interactions_matrix'] = temp_pred['interactions_matrix']
        except Exception as e:
             print(f"[ERROR] Falló al reconstruir dependencias de preproceso para {args.model_name}: {e}")
             return
    
    # Cargar el objeto/estado del modelo
    try:
        if args.model_name in ['svd_model', 'item_knn_model', 'user_knn_model']:
            model = load_surprise_model(model_path)
        elif args.model_name in ['lightfm_model', 'als_model']:
            model = load_pickle_model(model_path, args.model_name)
        elif args.model_name == 'ncf_model':
             model = load_pytorch_model(NCF, model_path, user_map, item_map, embedding_dim=64, mlp_layers=[64, 32, 16])
        elif args.model_name == 'multivae_model':
             model = load_pytorch_model(MultiVAE, model_path, user_map, item_map, hidden_dim=600, latent_dim=200)
        elif args.model_name == 'lightgcn_model':
             model = load_pytorch_model(LightGCN, model_path, user_map, item_map, num_layers=3, embedding_dim=64, **kwargs)
        elif args.model_name == 'most_popular_model':
             model = load_most_popular(model_path) # Retorna (scores, avg)
        elif args.model_name == 'random_model':
             model = {'min_rating': 1.0, 'max_rating': 5.0} # Placeholder
        else:
            print(f"[ERROR] Lógica de carga no definida para: {args.model_name}")
            return
    except Exception as e:
        print(f"[ERROR] Falló la carga del modelo: {e}")
        return

    if model is None:
        print("[ERROR] Carga del modelo fallida, abortando.")
        return

    # --- 5. Preparar Datos de Predicción (Antitest del Usuario) ---
    try:
        antitest_file = os.path.join(data_path, 'antitest.csv')
        full_antitest_df = pd.read_csv(antitest_file)
        # Filtrar solo para el usuario de interés
        user_antitest_df = full_antitest_df[full_antitest_df['userId'] == args.user_id].copy()
        if user_antitest_df.empty:
            print(f"[ERROR] Usuario {args.user_id} no tiene ítems en el archivo antitest.")
            return
        print(f"   Se van a rankear {len(user_antitest_df)} ítems para el usuario {args.user_id}.")
    except Exception as e:
        print(f"[ERROR] No se pudo cargar o filtrar el archivo antitest: {e}")
        return
        
    # --- 6. Medir Inferencia Top-10 ---
    print("   Iniciando tracker de CodeCarbon para la predicción...")
    prediction_tracker = EmissionsTracker(log_level='error', save_to_file=False) # No guardar log de tracker
    prediction_emissions = None
    
    try:
        prediction_tracker.start()
        
        # Preparar el diccionario de componentes
        prediction_components_user = {
            'antitest_df': user_antitest_df, # Pasar el DF filtrado
            'user_map': user_map,
            'item_map': item_map
        }
        # Añadir dependencias especiales
        if args.model_name == 'lightgcn_model':
            prediction_components_user['edge_index'] = kwargs.get('norm_adj_matrix') # Renombrar clave si es necesario
        if args.model_name == 'multivae_model':
            prediction_components_user['interactions_matrix'] = kwargs.get('interactions_matrix')

        # Llamar a la función generate_predictions del modelo
        # Usamos * para desempaquetar la tupla de args
        predictions_df = model_module.generate_predictions(model, prediction_components_user)

        # Detener el tracker
        prediction_emissions = prediction_tracker.stop()
        print("   Tracker de CodeCarbon detenido.")

        if predictions_df is None or predictions_df.empty:
            print("[ERROR] La función generate_predictions no devolvió resultados.")
            return
            
        # --- 7. Procesar Resultados ---
        # Ordenar por predicción y tomar Top 10
        top_10_df = predictions_df.sort_values(by='prediction', ascending=False).head(10)
        top_10_movie_ids = top_10_df['movieId'].tolist() # Guardar los IDs originales

        print("\n--- Top 10 Recomendaciones ---")
        print(top_10_df)
        
        # --- 8. Guardar Resultados ---
        save_individual_results(results_file, args.dataset_percentage, args.model_name, args.user_id, top_10_movie_ids, prediction_emissions)

    except Exception as e:
        print(f"[ERROR] Falló la generación de predicciones Top-10: {e}")
        import traceback
        traceback.print_exc()
        if prediction_tracker._running:
            prediction_tracker.stop()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar Top-10 recs para un usuario y medir huella de carbono.")
    parser.add_argument("--model_name", type=str, required=True, help="Nombre del modelo (ej: svd_model)")
    parser.add_argument("--dataset_percentage", type=str, required=True, choices=['10', '25', '50', '75', '100'], help="Porcentaje del dataset")
    parser.add_argument("--user_id", type=int, required=True, help="ID del usuario para el que se generará el Top-10")

    args = parser.parse_args()
    main(args)