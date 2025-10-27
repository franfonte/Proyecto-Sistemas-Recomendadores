import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import pickle
import random # Para fijar semilla en modelo Random
import importlib 

# Importaciones específicas de librerías
import torch
from torch.utils.data import DataLoader
from surprise.dump import load as surprise_load
from lightfm import LightFM # Necesario para cargar pickle
import implicit # Necesario para cargar pickle de ALS
from codecarbon import EmissionsTracker # Importar CodeCarbon
from scipy.sparse import csr_matrix 

# Importar Clases de Modelos PyTorch (Necesario para cargar state_dict)
# Asumimos que los scripts están en models/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))
try:
    from ncf_model import NCF
    from lightgcn_model import LightGCN
    from multivae_model import MultiVAE
except ImportError as e:
    print(f"Advertencia: No se pudieron importar todas las clases de modelos PyTorch: {e}")
    # Definir placeholders si no se importan para evitar NameError más tarde
    class NCF: pass
    class LightGCN: pass
    class MultiVAE: pass


# --- Constante para Replicabilidad ---
RANDOM_SEED = 42

# --- FUNCIÓN para cargar mapas JSON ---
def load_maps_from_json(model_dir, model_name):
    """Carga user_map y item_map desde archivos JSON."""
    user_map_path = os.path.join(model_dir, f"{model_name}_user_map.json")
    item_map_path = os.path.join(model_dir, f"{model_name}_item_map.json")
    user_map, item_map = None, None
    try:
        with open(user_map_path, 'r') as f:
            user_map_str_keys = json.load(f)
            # Convertir claves a int (asumiendo que userId es int)
            user_map = {int(k): v for k, v in user_map_str_keys.items()}
        with open(item_map_path, 'r') as f:
            item_map_str_keys = json.load(f)
            # Convertir claves a int (asumiendo que movieId es int)
            item_map = {int(k): v for k, v in item_map_str_keys.items()}
        print("   Mapas de ID cargados desde JSON.")
    except FileNotFoundError:
        print(f"   Advertencia: Archivos de mapa no encontrados para {model_name}.")
    except Exception as e:
        print(f"   [ERROR] Falló al cargar mapas JSON: {e}")
    return user_map, item_map

# --- Función Helper para guardar JSON (copiada de run_experiment_saves) ---
def save_json_with_numpy(data, filepath):
    """Guarda un diccionario (posiblemente con tipos numpy) como JSON."""
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    def convert_keys_to_string(obj):
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, (int, np.integer, np.int64)) else k: convert_keys_to_string(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_keys_to_string(i) for i in obj]
        return obj

    data_to_save = convert_keys_to_string(data)
    with open(filepath, 'w') as f:
        json.dump(data_to_save, f, indent=4, cls=NumpyEncoder)
    print(f"   Resultados individuales guardados en: {filepath}")

# --- Funciones de Carga de Modelos ---
# (Ligeramente modificadas para pasar hiperparámetros)
def load_pytorch_model(model_class, state_dict_path, user_map, item_map, **kwargs):
    """Carga un modelo PyTorch."""
    num_users = len(user_map) if user_map else 0
    num_items = len(item_map) if item_map else 0
    
    model_instance = None
    try:
        if model_class == MultiVAE:
             if not num_items:
                  print("[ERROR] Se necesita num_items para instanciar MultiVAE.")
                  return None
             # Pasa los hiperparámetros correctos (del archivo multivae_model.py)
             model_instance = model_class(
                 num_items=num_items, 
                 hidden_dim=kwargs.get('hidden_dim', 600), 
                 latent_dim=kwargs.get('latent_dim', 200)
             )
        elif model_class == NCF:
              if not num_users or not num_items:
                  print("[ERROR] Se necesita num_users y num_items para instanciar NCF.")
                  return None
              # Pasa los hiperparámetros correctos (del archivo ncf_model.py)
              model_instance = model_class(
                  num_users=num_users, 
                  num_items=num_items, 
                  embedding_dim=kwargs.get('embedding_dim', 64), 
                  mlp_layers=kwargs.get('mlp_layers', [64, 32, 16])
              )
        elif model_class == LightGCN:
             print("[ERROR] Carga individual para LightGCN no implementada.")
             return None
        
        if model_instance:
            model_instance.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
            model_instance.eval()
            print(f"   Modelo {model_class.__name__} cargado correctamente.")
    except FileNotFoundError:
         print(f"[ERROR] Archivo state_dict no encontrado en: {state_dict_path}")
    except Exception as e:
        print(f"[ERROR] Falló al instanciar o cargar state_dict de {model_class.__name__}: {e}")
    return model_instance

def load_surprise_model(model_path):
    """Carga un modelo Surprise."""
    try:
        _, model = surprise_load(model_path)
        print("   Modelo Surprise cargado.")
        return model
    except Exception as e: print(f"[ERROR] Falló al cargar modelo Surprise: {e}")
    return None

def load_pickle_model(model_path, model_name):
    """Carga un modelo guardado con Pickle (LightFM, ALS)."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"   Modelo {model_name} (pickle) cargado.")
        return model
    except Exception as e: print(f"[ERROR] Falló al cargar modelo Pickle ({model_name}): {e}")
    return None

def load_most_popular(model_path):
    """Carga el 'modelo' Most Popular (scores y promedio global)."""
    try:
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        scores_dict = model_data.get('popularity_scores', {})
        popularity_scores = {int(k): v for k, v in scores_dict.items()}
        global_average = model_data.get('global_average')
        if global_average is None:
             print(f"[ERROR] No se encontró 'global_average' en {model_path}")
             return None, None
        print("   Modelo Most Popular cargado.")
        return pd.Series(popularity_scores), global_average
    except Exception as e:
        print(f"[ERROR] Falló al cargar modelo Most Popular: {e}")
    return None, None

# --- Función Principal (Lógica de Predicción Individual) ---
def get_single_top10(model_name, model, data_path, user_map, item_map, user_id):
    """
    Función central para generar predicciones Top-10 para un usuario.
    Esta función es la que será medida por CodeCarbon.
    """
    print(f"   Generando predicciones para usuario {user_id}...")
    
    # 1. Obtener el índice del usuario
    user_idx = user_map.get(user_id) if user_map else user_id # Usar ID directo si no hay mapa
    if user_idx is None:
        print("   [INFO] Usuario no encontrado en el mapa. No se puede predecir.")
        return pd.DataFrame(), None # Devolver DF vacío y sin datos de emisión

    # 2. Obtener todos los IDs de ítems posibles
    if item_map:
        all_item_indices = list(item_map.values())
        all_item_raw_ids = list(item_map.keys())
    else: # Para Surprise, necesitamos IDs raw
        train_file = os.path.join(data_path, 'train.csv')
        if not os.path.exists(train_file):
            print(f"   [ERROR] No se encontró {train_file} para obtener la lista de ítems.")
            return pd.DataFrame(), None
        train_df = pd.read_csv(train_file)
        all_item_raw_ids = train_df['movieId'].unique()
        all_item_indices = all_item_raw_ids
        
    num_all_items = len(all_item_raw_ids)
    
    # 3. Generar predicciones para todos los ítems para este usuario
    predictions = []
    
    tracker = EmissionsTracker(log_level='error')
    emissions_data = None
    
    try:
        if model_name in ['svd_model', 'item_knn_model', 'user_knn_model']:
            # Lógica para Surprise: predecir para todos los ítems raw
            tracker.start()
            for i, item_raw_id in enumerate(all_item_raw_ids):
                pred = model.predict(uid=user_id, iid=item_raw_id)
                predictions.append({'movieId_raw': item_raw_id, 'score': pred.est})
            emissions = tracker.stop()
            emissions_data = getattr(tracker, 'final_emissions_data', None)
                
        elif model_name in ['lightfm_model', 'als_model']:
            # Lógica para LightFM/ALS: predecir para todos los índices
            user_idx_array = np.array([user_idx] * num_all_items, dtype=np.int32)
            item_indices_array = np.array(all_item_indices, dtype=np.int32)
            
            if model_name == 'als_model':
                num_users_model = model.user_factors.shape[0]
                num_items_model = model.item_factors.shape[0]
                if user_idx >= num_users_model:
                    print(f"   [INFO] user_idx {user_idx} fuera de rango para factores ALS. No se puede predecir.")
                    return pd.DataFrame(), None
                valid_mask = item_indices_array < num_items_model
                item_indices_array = item_indices_array[valid_mask]
                all_item_raw_ids = np.array(all_item_raw_ids)[valid_mask]
                user_idx_array = user_idx_array[valid_mask]
            
            tracker.start()
            if model_name == 'lightfm_model':
                scores = model.predict(user_idx_array, item_indices_array, num_threads=1)
            else: # ALS
                user_factor = model.user_factors[user_idx]
                item_factors_batch = model.item_factors[item_indices_array]
                scores = np.dot(item_factors_batch, user_factor)
            emissions = tracker.stop()
            emissions_data = getattr(tracker, 'final_emissions_data', None)
                
            predictions = [{'movieId_raw': raw_id, 'score': score} for raw_id, score in zip(all_item_raw_ids, scores)]

        elif model_name in ['ncf_model', 'multivae_model']:
            device = torch.device("cpu")
            model.to(device)
            
            if model_name == 'ncf_model':
                pred_loader = DataLoader(list(zip([user_idx] * num_all_items, all_item_indices)), batch_size=1024, shuffle=False)
                scores = []
                tracker.start()
                with torch.no_grad():
                    for batch in pred_loader:
                        user_tensor = batch[0].long().to(device)
                        item_tensor = batch[1].long().to(device)
                        scores.extend(model(user_tensor, item_tensor).cpu().numpy().tolist())
                emissions = tracker.stop()
                emissions_data = getattr(tracker, 'final_emissions_data', None)
                predictions = [{'movieId_raw': raw_id, 'score': score} for raw_id, score in zip(all_item_raw_ids, scores)]
            
            elif model_name == 'multivae_model':
                # --- Preparación (Fuera del tracker) ---
                train_file = os.path.join(data_path, 'train.csv')
                train_df = pd.read_csv(train_file)
                train_df.loc[:, 'rating_bin'] = train_df['rating'].apply(lambda x: 1 if x >= 4.0 else 0)
                train_positive = train_df[train_df['rating_bin'] == 1].copy() # .copy() para evitar warning
                train_positive.loc[:, 'user_idx_map'] = train_positive['userId'].map(user_map)
                train_positive.loc[:, 'item_idx_map'] = train_positive['movieId'].map(item_map)
                train_positive = train_positive.dropna(subset=['user_idx_map', 'item_idx_map'])
                train_positive['user_idx_map'] = train_positive['user_idx_map'].astype(int)
                train_positive['item_idx_map'] = train_positive['item_idx_map'].astype(int)
                num_users = len(user_map)
                num_items = len(item_map)
                interactions_matrix = csr_matrix(
                     (np.ones(len(train_positive)), (train_positive['user_idx_map'], train_positive['item_idx_map'])),
                     shape=(num_users, num_items), dtype=np.float32)
                user_history = torch.FloatTensor(interactions_matrix[user_idx].toarray()).to(device)
                # --- Fin Preparación ---

                tracker.start()
                with torch.no_grad():
                    recon_logits, _, _ = model(user_history)
                    log_probs = torch.log_softmax(recon_logits, dim=-1).squeeze()
                    scores = log_probs.cpu().numpy()
                emissions = tracker.stop()
                emissions_data = getattr(tracker, 'final_emissions_data', None)
                    
                predictions = [{'movieId_raw': raw_id, 'score': scores[idx]} for raw_id, idx in item_map.items()]

        elif model_name == 'most_popular_model':
            popularity_scores, global_average = model
            tracker.start()
            for item_raw_id in all_item_raw_ids:
                score = popularity_scores.get(item_raw_id, global_average)
                predictions.append({'movieId_raw': item_raw_id, 'score': score})
            emissions = tracker.stop()
            emissions_data = getattr(tracker, 'final_emissions_data', None)
                
        elif model_name == 'random_model':
            random_gen = np.random.RandomState(RANDOM_SEED)
            tracker.start()
            scores = random_gen.uniform(1.0, 5.0, num_all_items)
            emissions = tracker.stop()
            emissions_data = getattr(tracker, 'final_emissions_data', None)
            predictions = [{'movieId_raw': raw_id, 'score': score} for raw_id, score in zip(all_item_raw_ids, scores)]
            
        else:
            print(f"   [ERROR] Lógica de predicción Top-10 no definida para: {model_name}")
            return pd.DataFrame(), None

    except Exception as e:
        print(f"   [ERROR] Falló la predicción individual para {model_name}: {e}")
        emissions = tracker.stop() # Intentar detener si falló
        emissions_data = getattr(tracker, 'final_emissions_data', None)
        return pd.DataFrame(), emissions_data # Devolver datos de emisión parciales si existen

    # 4. Ordenar y obtener Top-10
    if not predictions:
        print("   [ERROR] No se generó ninguna predicción.")
        return pd.DataFrame(), emissions_data
        
    pred_df = pd.DataFrame(predictions)
    
    # <<<<<<< CAMBIO CRÍTICO: Filtrar ítems ya vistos (train) >>>>>>>
    try:
        # Cargar historial de train del usuario
        train_file = os.path.join(data_path, 'train.csv')
        if not os.path.exists(train_file):
            print(f"   [ADVERTENCIA] No se encontró {train_file}, no se pueden filtrar ítems ya vistos.")
            seen_items = set()
        else:
            # Cargar solo las filas del usuario relevante
            # Leer en chunks si el archivo es muy grande
            chunk_iter = pd.read_csv(train_file, usecols=['userId', 'movieId'], chunksize=100000)
            user_train_df = pd.concat([chunk[chunk['userId'] == user_id] for chunk in chunk_iter])
            seen_items = set(user_train_df['movieId'].unique())
            print(f"   Se encontraron {len(seen_items)} ítems en el historial de train del usuario {user_id} para filtrar.")
            
        # Filtrar el DataFrame de predicciones
        pred_df_unseen = pred_df[~pred_df['movieId_raw'].isin(seen_items)]
        print(f"   Predicciones reducidas de {len(pred_df)} a {len(pred_df_unseen)} después de filtrar ítems vistos.")
        
        if pred_df_unseen.empty:
            print("   [ADVERTENCIA] No quedaron ítems después de filtrar los ya vistos.")
            return pd.DataFrame(), emissions_data
            
        top_10_df = pred_df_unseen.nlargest(10, 'score')
        
    except Exception as e:
        print(f"   [ERROR] Falló al filtrar ítems ya vistos: {e}")
        # Fallback: usar el top-10 sin filtrar
        top_10_df = pred_df.nlargest(10, 'score')
    # --- Fin del cambio ---
    
    print(f"   Top-10 generado exitosamente para usuario {user_id}.")
    return top_10_df, emissions_data # <<<<<<< CAMBIO AQUÍ: Devolver datos de emisión

# --- Función Principal de Ejecución ---
def main(args):
    # <<<<<<< CAMBIO AQUÍ: Capturar la nueva variable >>>>>>>
    user_id = args.user_id
    user_category = args.user_category
    
    print(f"--- Generando Top-10 para Usuario: {user_id} ({user_category}) ---")
    print(f"Modelo: {args.model_name}, Dataset: {args.dataset_percentage}%")

    # --- 1. Determinar Rutas ---
    data_path = os.path.join('data', args.dataset_percentage)
    model_dir = os.path.join('trained_models', args.dataset_percentage)
    
    extension = '.pkl'
    is_pytorch_model = args.model_name in ['ncf_model', 'lightgcn_model', 'multivae_model']
    if is_pytorch_model: extension = '.pth'
    elif args.model_name == 'most_popular_model': extension = '.json'
    model_path = os.path.join(model_dir, f"{args.model_name}{extension}")

    RESULTS_JSON_FILE = 'individual_results.json'

    # --- 2. Cargar Mapas (si es necesario) ---
    user_map, item_map = None, None
    needs_maps = args.model_name not in ['svd_model', 'item_knn_model', 'user_knn_model', 'most_popular_model', 'random_model']
    if needs_maps:
        user_map, item_map = load_maps_from_json(model_dir, args.model_name)
        if user_map is None or item_map is None:
            print("[ERROR] No se pudieron cargar los mapas necesarios para este modelo.")
            return

    # --- 3. Cargar Modelo ---
    model = None
    try:
        if args.model_name in ['svd_model', 'item_knn_model', 'user_knn_model']:
            model = load_surprise_model(model_path)
        elif args.model_name in ['lightfm_model', 'als_model']:
            model = load_pickle_model(model_path, args.model_name)
        elif args.model_name == 'ncf_model':
             model = load_pytorch_model(NCF, model_path, user_map, item_map,
                                        embedding_dim=64, mlp_layers=[64, 32, 16])
        elif args.model_name == 'multivae_model':
             model = load_pytorch_model(MultiVAE, model_path, user_map, item_map,
                                        hidden_dim=600, latent_dim=200)
        elif args.model_name == 'lightgcn_model':
             print("[ERROR] Carga individual de LightGCN no está soportada. Omitiendo.")
             return
        elif args.model_name == 'most_popular_model':
             model = load_most_popular(model_path)
        elif args.model_name == 'random_model':
             model = {'min_rating': 1.0, 'max_rating': 5.0}
        else:
            print(f"[ERROR] Lógica de carga no definida para: {args.model_name}")
            return
            
        if model is None:
            print(f"[ERROR] Falló la carga del modelo '{args.model_name}'.")
            return
            
    except Exception as e:
        print(f"[ERROR] Falló al cargar o instanciar el modelo: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 4. Medir Inferencia Individual ---
    print("\n--- MIDIENDO INFERENCIA INDIVIDUAL ---")
    
    try:
        # Llamar a la función que ahora contiene el tracker
        top_10_df, emissions_data = get_single_top10(
            model_name=args.model_name,
            model=model,
            data_path=data_path,
            user_map=user_map,
            item_map=item_map,
            user_id=user_id # Usar la variable capturada
        )
    except Exception as e:
        print(f"[ERROR] Falló la ejecución de get_single_top10: {e}")
        import traceback
        traceback.print_exc()
        top_10_df, emissions_data = pd.DataFrame(), None # Asegurar que existan
    
    print("--- MEDICIÓN FINALIZADA ---")

    # --- 5. Recopilar y Guardar Resultados ---
    if top_10_df.empty:
        print("No se generó Top-10. No se guardarán resultados.")
        return
        
    top_10_list = top_10_df['movieId_raw'].tolist()
    
    results_data = {
        "top_10_recommendations": top_10_list,
        "inference_footprint": {
            "co2_emissions_g": emissions_data.emissions * 1000 if emissions_data and emissions_data.emissions is not None else None,
            "energy_consumed_kWh": emissions_data.energy_consumed if emissions_data else None,
            "duration_seconds": emissions_data.duration if emissions_data else None
        }
    }
    
    # Cargar y actualizar el archivo JSON de resultados individuales
    if os.path.exists(RESULTS_JSON_FILE):
        with open(RESULTS_JSON_FILE, 'r') as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = {}
    else:
        all_results = {}

    dataset_key = str(args.dataset_percentage)
    # <<<<<<< CAMBIO AQUÍ: Crear la clave compuesta >>>>>>>
    user_key = f"{user_id} ({user_category})"
    
    all_results.setdefault(dataset_key, {}).setdefault(args.model_name, {})[user_key] = results_data
    
    try:
        save_json_with_numpy(all_results, RESULTS_JSON_FILE)
    except Exception as e:
        print(f"[ERROR] No se pudo guardar el archivo {RESULTS_JSON_FILE}: {e}")

    # --- 6. Reporte Final en Consola ---
    print("\n" + "="*60)
    # <<<<<<< CAMBIO AQUÍ: Mostrar la clave compuesta >>>>>>>
    print(f"REPORTE DE INFERENCIA INDIVIDUAL (Usuario: {user_key})")
    print("="*60)
    print(f"  Modelo: {args.model_name} (Dataset: {args.dataset_percentage}%)")
    print(f"  Top 10 (MovieIDs): {top_10_list}")
    if emissions_data:
        print(f"  Duración de Inferencia: {emissions_data.duration:.6f} segundos")
        print(f"  Energía Consumida: {emissions_data.energy_consumed:.10f} kWh")
        print(f"  Emisiones CO₂: {emissions_data.emissions * 1000:.10f} g")
    else:
        print("  No se pudieron obtener datos de CodeCarbon.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carga un modelo entrenado y genera un Top-10 para un usuario, midiendo la huella de carbono.")
    parser.add_argument("--model_name", type=str, required=True, help="Nombre del modelo (ej: svd_model)")
    parser.add_argument("--dataset_percentage", type=str, required=True, choices=['10', '25', '50', '75', '100'], help="Porcentaje del dataset")
    parser.add_argument("--user_id", type=int, required=True, help="ID del usuario (el ID original, ej. 5341)")
    # <<<<<<< CAMBIO AQUÍ: Añadir nuevo argumento >>>>>>>
    parser.add_argument("--user_category", type=str, default="General", help="Categoría del usuario (ej: Power User)")

    args = parser.parse_args()
    
    # Validar que las clases PyTorch estén disponibles si se necesitan
    if args.model_name in ['ncf_model', 'lightgcn_model', 'multivae_model']:
         if (args.model_name == 'ncf_model' and 'NCF' in globals() and NCF.__name__ == 'NCF') or \
            (args.model_name == 'multivae_model' and 'MultiVAE' in globals() and MultiVAE.__name__ == 'MultiVAE'):
              pass # OK
         elif (args.model_name == 'lightgcn_model'):
              print("[ERROR] Carga individual de LightGCN no está soportada.")
              sys.exit(1)
         else:
              # Esto puede pasar si la importación al inicio falló
              print("[ERROR] Clases PyTorch no importadas correctamente. Verifica la ruta y los archivos.")
              sys.exit(1)

    main(args)

