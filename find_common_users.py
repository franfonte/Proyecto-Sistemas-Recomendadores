import pandas as pd
import os
import sys
import json
import glob
from functools import reduce
import argparse # <<<<<<< CORRECCIÓN: Añadida la importación que faltaba

def find_common_users(data_path):
    """
    Encuentra la intersección de usuarios que tienen predicciones
    en TODOS los archivos *_predictions.csv dentro de un directorio.
    """
    print(f"Buscando usuarios comunes en: {data_path}")
    
    # Encontrar todos los archivos de predicciones
    pred_files = glob.glob(os.path.join(data_path, "*_predictions.csv"))
    
    if not pred_files:
        print(f"  [ERROR] No se encontraron archivos *_predictions.csv en {data_path}.")
        print("  Asegúrate de haber ejecutado run_experiment_saves.py primero.")
        return

    all_user_sets = []
    
    # Cargar los userIDs de cada archivo
    for f in pred_files:
        model_name = os.path.basename(f).replace("_predictions.csv", "")
        try:
            df = pd.read_csv(f)
            # Asegurarse de que la columna exista y no esté vacía
            if 'userId' in df.columns and not df.empty:
                user_set = set(df['userId'].unique())
                all_user_sets.append(user_set)
                print(f"  Encontrados {len(user_set)} usuarios únicos para el modelo: {model_name}")
            else:
                print(f"  Advertencia: Archivo {f} está vacío o no tiene columna 'userId'.")
        except pd.errors.EmptyDataError:
            print(f"  Advertencia: Archivo {f} está vacío.")
        except Exception as e:
            print(f"  [ERROR] No se pudo leer {f}: {e}")

    if not all_user_sets:
        print("  [ERROR] No se pudieron cargar conjuntos de usuarios válidos.")
        return

    # Encontrar la intersección (usuarios comunes a todos)
    common_users = reduce(lambda a, b: a.intersection(b), all_user_sets)
    
    if not common_users:
        print("  [ERROR] No se encontró ningún usuario común en todos los archivos de predicción.")
        return

    print(f"\nSe encontraron {len(common_users)} usuarios comunes en todos los modelos.")
    
    # Guardar la lista de usuarios comunes
    output_file = os.path.join(data_path, 'common_users.json')
    try:
        # Convertir set a lista para guardarlo en JSON
        save_json_with_numpy(list(common_users), output_file)
        print(f"Lista de usuarios comunes guardada en: {output_file}")
    except Exception as e:
        print(f"  [ERROR] No se pudo guardar el archivo JSON: {e}")

# (Reutilizamos la función de guardado de run_experiment_saves.py)
def save_json_with_numpy(data, filepath):
    """Guarda datos (incluyendo tipos numpy) como JSON."""
    # Necesitamos importar numpy aquí si esta función se va a usar
    import numpy as np 
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encontrar usuarios comunes en todos los archivos de predicción de un dataset.")
    parser.add_argument("--dataset_percentage", type=str, required=True, choices=['10', '25', '50', '75', '100'], help="Porcentaje del dataset (ej. 10)")
    args = parser.parse_args()
    
    data_path = os.path.join('data', args.dataset_percentage)
    if not os.path.exists(data_path):
        print(f"[ERROR] El directorio de datos no existe: {data_path}")
        sys.exit(1)
        
    find_common_users(data_path)