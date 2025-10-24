import os
import argparse
import sys
import importlib
import json
import pandas as pd
from codecarbon import EmissionsTracker
from evaluate_results import calculate_rmse, calculate_ranking_metrics

# --- NUEVAS IMPORTACIONES PARA GUARDAR MODELOS Y MAPAS ---
import torch
from surprise.dump import dump as surprise_dump
import pickle
import numpy as np # Para manejar tipos al guardar JSON

# --- FUNCIÓN HELPER para guardar JSON (maneja tipos numpy y claves int) ---
def save_json_with_numpy(data, filepath):
    """Guarda un diccionario (posiblemente con tipos numpy) como JSON.
       Convierte claves int/int64 a string."""
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64)): # Añadido np.int64
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)): # Añadido float types
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    # Convertir claves a string recursivamente
    def convert_keys_to_string(obj):
        if isinstance(obj, dict):
            # Convertir clave si es necesario, luego aplicar recursivamente
            return {str(k) if isinstance(k, (int, np.integer, np.int64)) else k: convert_keys_to_string(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_keys_to_string(i) for i in obj]
        return obj

    data_str_keys = convert_keys_to_string(data)

    with open(filepath, 'w') as f:
        # Usar el encoder personalizado para valores numpy
        json.dump(data_str_keys, f, indent=4, cls=NumpyEncoder)
    print(f"   Mapa/Dato guardado en: {filepath}")


def update_results_json(filepath, dataset_percentage, model_name, results_data):
    """
    Carga un archivo JSON, actualiza los datos de un experimento y lo guarda.
    Crea el archivo si no existe.
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
     # Crear diccionarios anidados si no existen
    if dataset_key not in all_results:
        all_results[dataset_key] = {}
    if model_name not in all_results[dataset_key]:
        all_results[dataset_key][model_name] = {}

    # Actualizar/Añadir los datos del experimento actual
    all_results[dataset_key][model_name] = results_data


    # Convertir NaN a None antes de guardar para compatibilidad JSON
    def convert_nan_to_none(obj):
        if isinstance(obj, dict):
            return {k: convert_nan_to_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_nan_to_none(i) for i in obj]
        # Usar pd.isna() para chequear NaN de forma segura (cubre np.nan y pd.NA)
        if isinstance(obj, float) and pd.isna(obj):
            return None
        return obj
    all_results = convert_nan_to_none(all_results)


    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=4, sort_keys=True)

    print(f"\nResultados guardados y actualizados en '{filepath}'")


if __name__ == "__main__":
    # --- 1. Configuración de Argumentos ---
    parser = argparse.ArgumentParser(
        description="Run a recommender system model, measure its carbon footprint, and log results."
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model script name.")
    parser.add_argument("--dataset_percentage", type=str, required=True, choices=['10', '25', '50', '75', '100'], help="Dataset percentage.")
    args = parser.parse_args()

    # --- 2. Preparación de Rutas y Módulo ---
    DATA_PATH = os.path.join('data', args.dataset_percentage)
    JSON_RESULTS_FILE = 'results.json'
    MODEL_SAVE_DIR = os.path.join('trained_models', args.dataset_percentage)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Definir extensión del modelo
    is_pytorch_model = args.model_name in ['ncf_model', 'lightgcn_model', 'multivae_model']
    extension = '.pth' if is_pytorch_model else '.pkl'
    if args.model_name == 'most_popular_model': extension = '.json'
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"{args.model_name}{extension}")

    USER_MAP_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"{args.model_name}_user_map.json")
    ITEM_MAP_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"{args.model_name}_item_map.json")


    if not os.path.exists(DATA_PATH):
        print(f"Error: Data directory not found at '{DATA_PATH}'. Run 'prepare_datasets.py' first.")
        sys.exit(1)

    try:
        model_module = importlib.import_module(f"models.{args.model_name}")
    except ImportError as e:
        print(f"Error: Could not find module 'models/{args.model_name}.py'. {e}")
        sys.exit(1)

    # --- 3. Preprocesamiento de Datos ---
    print("\n--- PREPROCESSING DATA ---")
    try:
        preprocess_output = model_module.preprocess_data(DATA_PATH)
    except Exception as e:
        print(f"[ERROR] Falló la ejecución de preprocess_data para {args.model_name}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    maps_available = False # Default
    user_map, item_map = None, None

    # <<<<<<< DESEMPAQUETADO CORRECTO DE 4 o 2 VALORES >>>>>>>
    if len(preprocess_output) == 4:
        training_data, prediction_extra_args_val, user_map, item_map = preprocess_output
        maps_available = True
        print("   Mapas de ID recibidos desde preprocess_data.")
    elif len(preprocess_output) == 2:
        training_data, prediction_extra_args_val = preprocess_output
        print("   (Modelo no devuelve mapas explícitos desde preprocess_data)")
    else:
        print(f"[ERROR] La función preprocess_data de {args.model_name} devolvió un número inesperado de elementos: {len(preprocess_output)}")
        sys.exit(1)

    # Asegurar que prediction_extra_args sea siempre una tupla para desempaquetar con *
    if isinstance(prediction_extra_args_val, dict) or isinstance(prediction_extra_args_val, list):
         prediction_extra_args_tuple = (prediction_extra_args_val,) # Empaquetar dict/list en tupla
    elif isinstance(prediction_extra_args_val, tuple):
         prediction_extra_args_tuple = prediction_extra_args_val # Ya es tupla
    else:
         # Si es un solo valor (ej. matriz, trainset), empaquetarlo
         prediction_extra_args_tuple = (prediction_extra_args_val,)


    # --- 4. Medición de la Fase de ENTRENAMIENTO ---
    print("\n--- MEASURING TRAINING PHASE ---")
    training_tracker = EmissionsTracker(log_level='error')
    trained_model = None # Inicializar
    training_emissions = None
    try:
        training_tracker.start()
        trained_model = model_module.train_model(training_data)
        training_emissions = training_tracker.stop()
    except Exception as e:
        print(f"[ERROR] Falló la ejecución de train_model para {args.model_name}: {e}")
        training_emissions = training_tracker.stop() # Intentar detener tracker igualmente
        import traceback
        traceback.print_exc()
        # Continuar para guardar resultados parciales si es posible, o salir
        # sys.exit(1) # Descomentar si se prefiere detener todo en caso de error de entrenamiento
    print("--- TRAINING MEASUREMENT FINISHED ---")


    # --- 4.5. Guardar el modelo entrenado y los mapas ---
    print(f"\n--- SAVING TRAINED MODEL & MAPS ---")
    # Guardar Modelo (solo si el entrenamiento fue exitoso)
    if trained_model is not None:
        try:
            if args.model_name == 'random_model':
                print("   (El modelo 'Random' no tiene un objeto de modelo para guardar)")
            elif is_pytorch_model:
                torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
                print(f"   Modelo PyTorch (state_dict) guardado en: {MODEL_SAVE_PATH}")
            elif args.model_name in ['svd_model', 'item_knn_model', 'user_knn_model']:
                surprise_dump(MODEL_SAVE_PATH, algo=trained_model)
                print(f"   Modelo Surprise guardado en: {MODEL_SAVE_PATH}")
            elif args.model_name in ['lightfm_model', 'als_model']:
                with open(MODEL_SAVE_PATH, 'wb') as f: pickle.dump(trained_model, f)
                print(f"   Modelo '{args.model_name}' (pickle) guardado en: {MODEL_SAVE_PATH}")
            elif args.model_name == 'most_popular_model':
                 pop_scores, global_avg = trained_model
                 model_data_to_save = {
                     'popularity_scores': pop_scores.to_dict(),
                     'global_average': global_avg
                 }
                 save_json_with_numpy(model_data_to_save, MODEL_SAVE_PATH)
            else:
                print(f"   Advertencia: No se ha definido una lógica de guardado para {args.model_name}")
        except Exception as e:
            print(f"   [ERROR] No se pudo guardar el modelo en {MODEL_SAVE_PATH}: {e}")
    else:
        print("   (Entrenamiento falló, no se guardó el modelo)")


    # Guardar Mapas si están disponibles (independientemente del éxito del entrenamiento)
    if maps_available and user_map is not None and item_map is not None:
        try:
            save_json_with_numpy(user_map, USER_MAP_SAVE_PATH)
            save_json_with_numpy(item_map, ITEM_MAP_SAVE_PATH)
        except Exception as e:
            print(f"   [ERROR] No se pudieron guardar los mapas JSON: {e}")
    elif maps_available:
         print(f"   Advertencia: preprocess_data devolvió 4 elementos pero los mapas eran None.")


    # --- 5. Medición de la Fase de PREDICCÍÓN ---
    print("\n--- MEASURING PREDICTION PHASE ---")
    prediction_tracker = EmissionsTracker(log_level='error')
    predictions_df = pd.DataFrame(columns=['userId', 'movieId', 'prediction']) # Default vacío
    prediction_emissions = None

    # Solo intentar predecir si el entrenamiento fue exitoso
    if trained_model is not None:
        try:
            prediction_tracker.start()

            # <<<<<<< CONSTRUCCIÓN CORRECTA DE ARGUMENTOS PARA generate_predictions >>>>>>>
            # Siempre pasamos una tupla de argumentos posicionales usando *
            if maps_available:
                # Si hay mapas, asumimos que el primer argumento original era el diccionario
                # Creamos un NUEVO diccionario que incluye los mapas
                if prediction_extra_args_tuple and isinstance(prediction_extra_args_tuple[0], dict):
                     final_prediction_components = prediction_extra_args_tuple[0].copy() # Copiar dict original
                else:
                     # Si no había dict (ej. LightFM devolvía solo antitest_df antes)
                     # Crear uno nuevo, asumiendo que el primer arg es antitest_df
                     final_prediction_components = {'antitest_df': prediction_extra_args_tuple[0]} if prediction_extra_args_tuple else {}

                final_prediction_components['user_map'] = user_map
                final_prediction_components['item_map'] = item_map
                # Pasar solo este diccionario como argumento
                final_prediction_args_tuple_for_call = (final_prediction_components,)
            else:
                # Si no hay mapas, pasar los argumentos originales tal cual
                final_prediction_args_tuple_for_call = prediction_extra_args_tuple

            # Llamar a generate_predictions desempaquetando la tupla de argumentos
            predictions_df = model_module.generate_predictions(trained_model, *final_prediction_args_tuple_for_call)
            prediction_emissions = prediction_tracker.stop()

        except Exception as e:
            print(f"[ERROR] Falló la ejecución de generate_predictions para {args.model_name}: {e}")
            prediction_emissions = prediction_tracker.stop() # Intentar detener tracker
            import traceback
            traceback.print_exc()
            # Continuar con DF vacío para guardar resultados parciales
    else:
        print("   (Entrenamiento falló, se omite la predicción)")

    print("--- PREDICTION MEASUREMENT FINISHED ---")


    # --- 6. Guardar Predicciones ---
    predictions_output_path = os.path.join(DATA_PATH, f"{args.model_name}_predictions.csv")
    if isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty:
        try:
            predictions_df.to_csv(predictions_output_path, index=False)
            print(f"\nPredictions saved to '{predictions_output_path}'")
        except Exception as e:
            print(f"\n[ERROR] No se pudo guardar el archivo de predicciones en {predictions_output_path}: {e}")
    elif isinstance(predictions_df, pd.DataFrame) and predictions_df.empty and trained_model is not None:
         print(f"\n[INFO] generate_predictions devolvió un DataFrame vacío. No se guardó {predictions_output_path}")
    # No imprimir error si trained_model fue None


    # --- 7. Evaluación de Métricas ---
    print("\n--- EVALUATING METRICS ---")
    rmse, precision, recall, ndcg = np.nan, np.nan, np.nan, np.nan # Valores por defecto
    test_file = os.path.join(DATA_PATH, 'test.csv')
    if os.path.exists(test_file):
        try:
            test_df = pd.read_csv(test_file)
            # Calcular métricas solo si hay predicciones válidas
            if isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty:
                merged_df_for_rmse = pd.merge(test_df, predictions_df, on=['userId', 'movieId'], how='inner')
                rmse = calculate_rmse(merged_df_for_rmse)
                precision, recall, ndcg = calculate_ranking_metrics(predictions_df, test_df, k=10)
            elif trained_model is not None: # Solo advertir si el entrenamiento tuvo éxito pero no hubo preds
                 print("   Advertencia: No hay predicciones válidas para calcular métricas.")

        except FileNotFoundError:
            print(f"   [ERROR] Archivo de test no encontrado en {test_file}. No se pueden calcular métricas.")
        except Exception as e:
            print(f"   [ERROR] Falló el cálculo de métricas: {e}")
    else:
        print(f"   [ERROR] Archivo de test no encontrado en {test_file}. No se pueden calcular métricas.")

    print("--- METRICS CALCULATION FINISHED ---")

    # --- 8. Recopilar y Guardar Resultados en JSON ---
    training_data_cc = getattr(training_tracker, 'final_emissions_data', None)
    prediction_data_cc = getattr(prediction_tracker, 'final_emissions_data', None)

    final_results = {
        "training_footprint": {
            "co2_emissions_g": training_emissions * 1000 if isinstance(training_emissions, (int, float)) else None,
            "energy_consumed_kWh": training_data_cc.energy_consumed if training_data_cc else None,
            "duration_seconds": training_data_cc.duration if training_data_cc else None
        },
        "prediction_footprint": {
            "co2_emissions_g": prediction_emissions * 1000 if isinstance(prediction_emissions, (int, float)) else None,
            "energy_consumed_kWh": prediction_data_cc.energy_consumed if prediction_data_cc else None,
            "duration_seconds": prediction_data_cc.duration if prediction_data_cc else None
        },
        "performance_metrics": {
            "rmse": rmse,
            "precision_at_10": precision,
            "recall_at_10": recall,
            "ndcg_at_10": ndcg
        }
    }

    update_results_json(JSON_RESULTS_FILE, args.dataset_percentage, args.model_name, final_results)

    # --- 9. Reporte Final en Consola ---
    print("\n" + "="*60)
    print(f"FINAL REPORT: {args.model_name} on {args.dataset_percentage}% dataset")
    print("="*60)
    perf_metrics = final_results['performance_metrics']
    train_fp = final_results['training_footprint']
    pred_fp = final_results['prediction_footprint']

    # Función helper para formatear o mostrar N/A
    def format_metric(value, format_str=".4f"):
        if value is not None and not pd.isna(value):
            return f"{value:{format_str}}"
        return "N/A"

    print(f"  - RMSE              : {format_metric(perf_metrics.get('rmse'))}")
    print(f"  - Precision@10      : {format_metric(perf_metrics.get('precision_at_10'))}")
    print(f"  - Recall@10         : {format_metric(perf_metrics.get('recall_at_10'))}")
    print(f"  - nDCG@10           : {format_metric(perf_metrics.get('ndcg_at_10'))}")
    print(f"  - Training CO₂ (g)  : {format_metric(train_fp.get('co2_emissions_g'))}")
    print(f"  - Prediction CO₂ (g): {format_metric(pred_fp.get('co2_emissions_g'))}")
    print(f"  - Training Time (s) : {format_metric(train_fp.get('duration_seconds'), '.2f')}")
    print(f"  - Prediction Time (s): {format_metric(pred_fp.get('duration_seconds'), '.2f')}")