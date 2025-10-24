import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import pickle
import random # Para fijar semilla en modelo Random

# Importaciones específicas de librerías
import torch
from surprise.dump import load as surprise_load
from lightfm import LightFM # Necesario para cargar pickle
import implicit # Necesario para cargar pickle de ALS

# Importar Clases de Modelos PyTorch (Necesario para cargar state_dict)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))
try:
    # Asegúrate de que los nombres de archivo y clases coincidan
    from ncf_model import NCF
    from lightgcn_model import LightGCN # Aunque no se usa, mantener import si existe
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


# --- Funciones de Carga de Modelos ---
# (Sin cambios respecto a la versión anterior)
def load_pytorch_model(model_class, state_dict_path, user_map, item_map, **kwargs):
    """Carga un modelo PyTorch."""
    num_users = len(user_map) if user_map else 0
    num_items = len(item_map) if item_map else 0
    if not num_users or not num_items:
         # MultiVAE solo necesita num_items
         if model_class != MultiVAE and model_class != LightGCN : # LightGCN no se maneja aquí
              print("[ERROR] Mapas no disponibles para determinar dimensiones del modelo PyTorch.")
              return None
         elif model_class == MultiVAE and not num_items:
              print("[ERROR] Mapa de items no disponible para determinar dimensiones de MultiVAE.")
              return None


    model_instance = None
    try:
        # Instanciar modelo basado en la clase
        if model_class == MultiVAE:
             if not num_items: # Doble chequeo por si acaso
                  print("[ERROR] Se necesita num_items para instanciar MultiVAE.")
                  return None
             model_instance = model_class(num_items=num_items, **kwargs)
        elif model_class == NCF:
              if not num_users or not num_items: # Doble chequeo
                  print("[ERROR] Se necesita num_users y num_items para instanciar NCF.")
                  return None
              model_instance = model_class(num_users=num_users, num_items=num_items, **kwargs)
        elif model_class == LightGCN:
             print("[ERROR] Carga individual para LightGCN no implementada.")
             return None
        else:
             print(f"[ERROR] Clase de modelo PyTorch desconocida: {model_class}")
             return None

        # Cargar los pesos entrenados
        # Asegurarse de mapear a CPU por si se entrenó en GPU
        model_instance.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
        model_instance.eval()
        print(f"   Modelo {model_class.__name__} cargado correctamente.")
    except FileNotFoundError:
         print(f"[ERROR] Archivo state_dict no encontrado en: {state_dict_path}")
         return None
    except Exception as e:
        print(f"[ERROR] Falló al instanciar o cargar state_dict de {model_class.__name__}: {e}")
        return None
    return model_instance


def load_surprise_model(model_path):
    """Carga un modelo Surprise."""
    try:
        _, model = surprise_load(model_path)
        print("   Modelo Surprise cargado.")
        return model
    except FileNotFoundError:
         print(f"[ERROR] Archivo de modelo Surprise no encontrado en: {model_path}")
         return None
    except Exception as e:
        print(f"[ERROR] Falló al cargar modelo Surprise: {e}")
        return None

def load_pickle_model(model_path, model_name):
    """Carga un modelo guardado con Pickle (LightFM, ALS)."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"   Modelo {model_name} (pickle) cargado.")
        return model
    except FileNotFoundError:
        print(f"[ERROR] Archivo de modelo Pickle no encontrado en: {model_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Falló al cargar modelo Pickle ({model_name}): {e}")
        return None

def load_most_popular(model_path):
    """Carga el 'modelo' Most Popular (scores y promedio global)."""
    try:
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        # Convertir claves de scores a int (porque movieId es int)
        # Asegurarse de que 'popularity_scores' exista y sea un dict
        scores_dict = model_data.get('popularity_scores', {})
        if not isinstance(scores_dict, dict):
             print(f"[ERROR] Formato inesperado para 'popularity_scores' en {model_path}")
             return None, None
        popularity_scores = {int(k): v for k, v in scores_dict.items()}
        global_average = model_data.get('global_average')
        if global_average is None:
             print(f"[ERROR] No se encontró 'global_average' en {model_path}")
             return None, None
        print("   Modelo Most Popular cargado.")
        return pd.Series(popularity_scores), global_average # Devolver como Series
    except FileNotFoundError:
         print(f"[ERROR] Archivo de modelo Most Popular no encontrado en: {model_path}")
         return None, None
    except Exception as e:
        print(f"[ERROR] Falló al cargar modelo Most Popular: {e}")
        return None, None


# --- Función Principal (Modificada) ---
def main(args):
    print(f"--- Predicción Individual ---")
    print(f"Modelo: {args.model_name}, Dataset: {args.dataset_percentage}%")
    print(f"Usuario: {args.user_id}, Película: {args.movie_id}")

    # --- 1. Determinar Rutas ---
    data_path = os.path.join('data', args.dataset_percentage)
    model_dir = os.path.join('trained_models', args.dataset_percentage)
    batch_pred_file = os.path.join(data_path, f"{args.model_name}_predictions.csv")

    # Determinar extensión y ruta del modelo guardado
    extension = '.pkl'
    is_pytorch_model = args.model_name in ['ncf_model', 'lightgcn_model', 'multivae_model']
    if is_pytorch_model: extension = '.pth'
    elif args.model_name == 'most_popular_model': extension = '.json'
    model_path = os.path.join(model_dir, f"{args.model_name}{extension}")

    # <<<<<<< CAMBIO AQUÍ: Manejar caso 'random_model' ANTES de chequear archivo >>>>>>>
    if args.model_name == 'random_model':
        print("   (El modelo 'Random' no necesita cargar un archivo)")
        # Proceder directamente a la lógica de predicción/comparación
    elif not os.path.exists(model_path):
        # Si no es random Y el archivo no existe, es un error
        print(f"[ERROR] Archivo de modelo no encontrado en: {model_path}")
        return
    # --- Fin Cambio ---

    if not os.path.exists(batch_pred_file):
        # El archivo de predicciones en lote SÍ es necesario para la comparación
        print(f"[ERROR] Archivo de predicciones en lote no encontrado en: {batch_pred_file}")
        return

    # --- 2. Cargar Mapas (si es necesario) y Modelo ---
    model = None
    user_map, item_map = None, None
    prediction_single = None

    needs_maps = args.model_name not in ['svd_model', 'item_knn_model', 'user_knn_model', 'most_popular_model', 'random_model']
    if needs_maps:
        user_map, item_map = load_maps_from_json(model_dir, args.model_name)
        if user_map is None or item_map is None:
            print("[ERROR] No se pudieron cargar los mapas necesarios para este modelo.")
            # Permitir continuar solo si el modelo puede funcionar sin mapas (poco probable)
            # return # Descomentar si los mapas son estrictamente necesarios

    # --- Lógica de Carga y Predicción Específica ---
    try:
        if args.model_name in ['svd_model', 'item_knn_model', 'user_knn_model']:
            model = load_surprise_model(model_path)
            if model:
                pred_obj = model.predict(uid=args.user_id, iid=args.movie_id)
                prediction_single = pred_obj.est

        elif args.model_name == 'lightfm_model':
            model = load_pickle_model(model_path, args.model_name)
            if model and user_map is not None and item_map is not None:
                user_idx = user_map.get(args.user_id)
                item_idx = item_map.get(args.movie_id)
                if user_idx is not None and item_idx is not None:
                     prediction_single = model.predict(np.array([user_idx], dtype=np.int32),
                                                       np.array([item_idx], dtype=np.int32))[0]
                else: print("[INFO] Usuario o ítem no presente en los mapas de LightFM.")
            elif not model: print("[ERROR] Falló la carga del modelo LightFM.")
            else: print("[ERROR] Mapas necesarios para LightFM no cargados.")


        elif args.model_name == 'als_model':
             model = load_pickle_model(model_path, args.model_name)
             if model and user_map is not None and item_map is not None:
                 user_idx = user_map.get(args.user_id)
                 item_idx = item_map.get(args.movie_id)
                 # Verificar dimensiones directamente del modelo cargado
                 num_users_model = model.user_factors.shape[0]
                 num_items_model = model.item_factors.shape[0]
                 if user_idx is not None and item_idx is not None and user_idx < num_users_model and item_idx < num_items_model:
                     user_factor = model.user_factors[user_idx]
                     item_factor = model.item_factors[item_idx]
                     prediction_single = np.dot(user_factor, item_factor)
                 else: print("[INFO] Usuario o ítem no tiene factor aprendido en ALS.")
             elif not model: print("[ERROR] Falló la carga del modelo ALS.")
             else: print("[ERROR] Mapas necesarios para ALS no cargados.")


        elif args.model_name == 'ncf_model':
             if user_map is not None and item_map is not None:
                 # Pasar hiperparámetros consistentes con el entrenamiento
                 model = load_pytorch_model(NCF, model_path, user_map, item_map,
                                            embedding_dim=64, mlp_layers=[64, 32, 16]) # Asegúrate que coincidan
                 if model:
                     user_idx = user_map.get(args.user_id)
                     item_idx = item_map.get(args.movie_id)
                     if user_idx is not None and item_idx is not None:
                         with torch.no_grad():
                             user_tensor = torch.tensor([user_idx], dtype=torch.long)
                             item_tensor = torch.tensor([item_idx], dtype=torch.long)
                             # Mover tensores al dispositivo si es necesario (CPU por defecto aquí)
                             prediction_single = model(user_tensor, item_tensor).item()
                     else: print("[INFO] Usuario o ítem no presente en los mapas globales de NCF.")
                 # else: ya se imprimió error en load_pytorch_model
             else: print("[ERROR] Mapas necesarios para NCF no cargados.")


        elif args.model_name == 'multivae_model':
             if user_map is not None and item_map is not None:
                 # Pasar hiperparámetros consistentes
                 model = load_pytorch_model(MultiVAE, model_path, user_map, item_map,
                                            hidden_dim=600, latent_dim=200) # Asegúrate que coincidan
                 if model:
                     user_idx = user_map.get(args.user_id)
                     item_idx = item_map.get(args.movie_id)
                     if user_idx is not None and item_idx is not None:
                         # Reconstruir historial (simplificado, podría ser costoso si train.csv es grande)
                         try:
                             train_file = os.path.join(data_path, 'train.csv')
                             train_df = pd.read_csv(train_file)
                             train_df.loc[:, 'rating_bin'] = train_df['rating'].apply(lambda x: 1 if x >= 4.0 else 0)
                             train_positive = train_df[train_df['rating_bin'] == 1].copy()
                             train_positive.loc[:, 'user_idx_map'] = train_positive['userId'].map(user_map)
                             train_positive.loc[:, 'item_idx_map'] = train_positive['movieId'].map(item_map)
                             train_positive = train_positive.dropna(subset=['user_idx_map', 'item_idx_map'])
                             train_positive['user_idx_map'] = train_positive['user_idx_map'].astype(int)
                             train_positive['item_idx_map'] = train_positive['item_idx_map'].astype(int)
                             from scipy.sparse import csr_matrix
                             num_users = len(user_map)
                             num_items = len(item_map)
                             interactions_matrix = csr_matrix(
                                 (np.ones(len(train_positive)), (train_positive['user_idx_map'], train_positive['item_idx_map'])),
                                 shape=(num_users, num_items), dtype=np.float32)

                             with torch.no_grad():
                                 user_history = torch.FloatTensor(interactions_matrix[user_idx].toarray())
                                 recon_logits, _, _ = model(user_history) # Solo necesitamos logits
                                 log_probs = torch.log_softmax(recon_logits, dim=-1)
                                 # Asegurar que item_idx sea int
                                 prediction_single = log_probs[0, int(item_idx)].item()
                         except FileNotFoundError: print(f"[ERROR] No se encontró {train_file} para historial.")
                         except Exception as inner_e: print(f"[ERROR] Falló predicción MultiVAE: {inner_e}")
                     else: print("[INFO] Usuario o ítem no presente en los mapas globales de MultiVAE.")
                 # else: ya se imprimió error
             else: print("[ERROR] Mapas necesarios para MultiVAE no cargados.")


        elif args.model_name == 'lightgcn_model':
             print("[ERROR] Carga y predicción individual para LightGCN no implementada.")

        elif args.model_name == 'most_popular_model':
             popularity_scores, global_average = load_most_popular(model_path)
             if popularity_scores is not None:
                 # Necesita movieId original, que es int
                 prediction_single = popularity_scores.get(args.movie_id, global_average)

        elif args.model_name == 'random_model':
             # Generar predicción individual determinista
             random_gen = np.random.RandomState(RANDOM_SEED)
             prediction_single = random_gen.uniform(1.0, 5.0)
             print("[INFO] Predicción 'Random' individual generada.")

        else:
            print(f"[ERROR] Lógica de carga/predicción no definida para: {args.model_name}")
            return # Salir si el modelo no está manejado

    except Exception as e:
        print(f"[ERROR] Falló durante la carga o predicción específica del modelo: {e}")
        import traceback
        traceback.print_exc()
        # No retornar aquí, intentar obtener predicción de lote para comparación

    # --- 3. Obtener Predicción del Archivo CSV ---
    prediction_batch = None
    try:
        batch_df = pd.read_csv(batch_pred_file)
        batch_df['userId'] = batch_df['userId'].astype(int)
        batch_df['movieId'] = batch_df['movieId'].astype(int)
        match_row = batch_df[(batch_df['userId'] == args.user_id) & (batch_df['movieId'] == args.movie_id)]
        if not match_row.empty:
            prediction_batch = match_row['prediction'].iloc[0]
        elif args.model_name != 'random_model': # Para random es normal no encontrarlo si fue filtrado
            print(f"[INFO] El par ({args.user_id}, {args.movie_id}) no se encontró en {batch_pred_file}.")
            # No retornar todavía, comparar con prediction_single si existe

    except FileNotFoundError:
         print(f"[ERROR] Archivo de predicciones en lote no encontrado: {batch_pred_file}")
         # No podemos comparar, pero mostramos la predicción individual si se generó
    except Exception as e:
        print(f"[ERROR] Falló al leer o buscar en {batch_pred_file}: {e}")
        # No podemos comparar

    # --- 4. Comparar y Reportar ---
    print("\n--- Comparación ---")
    print(f"Predicción Individual : {prediction_single}")
    print(f"Predicción en Lote    : {prediction_batch}")

    if args.model_name == 'random_model':
        print("[INFO] Modelo 'Random': No se verifica coincidencia con lote.")
    elif prediction_single is not None and prediction_batch is not None:
        # Aumentar tolerancia ligeramente por posibles diferencias numéricas mínimas
        if np.isclose(prediction_single, prediction_batch, atol=1e-5):
            print("[ÉXITO] Las predicciones coinciden.")
        else:
            print("[FALLO] ¡Las predicciones NO coinciden!")
            print(f"        Diferencia: {abs(prediction_single - prediction_batch):.6f}")
    elif prediction_single is None and prediction_batch is None:
        print("[INFO] Ambas predicciones son Nulas/Desconocidas (esperado para ALS/LightFM si user/item no estaba en train).")
    # Considerar el caso donde uno es None y el otro no como fallo implícito
    elif prediction_batch is not None: # Si solo falta la individual
         print("[FALLO] No se pudo generar la predicción individual.")
    elif prediction_single is not None: # Si solo falta la de lote (ya se imprimió INFO arriba)
         print("[FALLO] Predicción individual generada, pero no encontrada en lote (posible filtro).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carga un modelo entrenado y genera una predicción individual.")
    parser.add_argument("--model_name", type=str, required=True, help="Nombre del modelo (ej: svd_model)")
    parser.add_argument("--dataset_percentage", type=str, required=True, choices=['10', '25', '50', '75', '100'], help="Porcentaje del dataset")
    parser.add_argument("--user_id", type=int, required=True, help="ID del usuario")
    parser.add_argument("--movie_id", type=int, required=True, help="ID de la película")
    args = parser.parse_args()

    # Validar importaciones PyTorch si es necesario
    if args.model_name in ['ncf_model', 'lightgcn_model', 'multivae_model']:
         # Verificar si las clases existen en el scope global
         if (args.model_name == 'ncf_model' and 'NCF' not in globals()) or \
            (args.model_name == 'lightgcn_model' and 'LightGCN' not in globals()) or \
            (args.model_name == 'multivae_model' and 'MultiVAE' not in globals()):
              # Intentar importar de nuevo por si acaso
              try:
                   from ncf_model import NCF
                   from lightgcn_model import LightGCN
                   from multivae_model import MultiVAE
              except ImportError:
                   print("[ERROR] Clases PyTorch no importadas correctamente. Verifica la ruta y los archivos.")
                   sys.exit(1)

    main(args)