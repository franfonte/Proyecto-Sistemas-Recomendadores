import os
import argparse
import sys
import importlib
import json
import pandas as pd
from codecarbon import EmissionsTracker
from evaluate_results import calculate_rmse, calculate_ranking_metrics

# --- IMPORTACIONES PARA GUARDAR MODELOS ---
import torch 
from surprise.dump import dump as surprise_dump
import pickle

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

    all_results.setdefault(dataset_percentage, {}).setdefault(model_name, results_data)
    all_results[dataset_percentage][model_name] = results_data

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

    # --- NUEVA SECCIÓN: Definir rutas de guardado de modelos ---
    MODEL_SAVE_DIR = os.path.join('trained_models', args.dataset_percentage)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Definir extensión (PyTorch vs. el resto)
    # <<<<<<< CAMBIO CRÍTICO AQUÍ: Se añade 'multivae_model' a la lista >>>>>>>
    is_pytorch_model = args.model_name in ['ncf_model', 'lightgcn_model', 'multivae_model']
    extension = '.pth' if is_pytorch_model else '.pkl'
    
    # Para el modelo 'most_popular', usaremos .json
    if args.model_name == 'most_popular_model':
        extension = '.json'
    
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"{args.model_name}{extension}")
    # --- Fin de la nueva sección ---

    if not os.path.exists(DATA_PATH):
        print(f"Error: Data directory not found at '{DATA_PATH}'. Run 'prepare_datasets.py' first.")
        sys.exit(1)

    try:
        model_module = importlib.import_module(f"models.{args.model_name}")
    except ImportError:
        print(f"Error: Could not find module 'models/{args.model_name}.py'.")
        sys.exit(1)

    # --- 3. Preprocesamiento de Datos ---
    print("\n--- PREPROCESSING DATA ---")
    preprocessed_data_tuple = model_module.preprocess_data(DATA_PATH)
    training_data = preprocessed_data_tuple[0]
    prediction_extra_args = preprocessed_data_tuple[1:]

    # --- 4. Medición de la Fase de ENTRENAMIENTO ---
    print("\n--- MEASURING TRAINING PHASE ---")
    training_tracker = EmissionsTracker(log_level='error')
    training_tracker.start()
    trained_model = model_module.train_model(training_data)
    training_emissions = training_tracker.stop()
    print("--- TRAINING MEASUREMENT FINISHED ---")

    # --- 4.5. Guardar el modelo entrenado ---
    print(f"\n--- SAVING TRAINED MODEL ---")
    try:
        if is_pytorch_model:
            # Guardar el state_dict de PyTorch
            torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
            print(f"   Modelo PyTorch (state_dict) guardado en: {MODEL_SAVE_PATH}")
        
        elif args.model_name in ['svd_model', 'item_knn_model', 'user_knn_model']:
            # Guardar con la utilidad de Surprise
            surprise_dump(MODEL_SAVE_PATH, algo=trained_model)
            print(f"   Modelo Surprise guardado en: {MODEL_SAVE_PATH}")

        # <<<<<<< CAMBIO AQUÍ: Añadida lógica para LightFM y ALS (ambos usan pickle) >>>>>>>
        elif args.model_name in ['lightfm_model', 'als_model']:
            # Guardar con pickle
            with open(MODEL_SAVE_PATH, 'wb') as f:
                pickle.dump(trained_model, f)
            print(f"   Modelo ({args.model_name}) guardado con pickle en: {MODEL_SAVE_PATH}")
        
        elif args.model_name == 'most_popular_model':
            # El "modelo" es una lista, guardar como JSON
            with open(MODEL_SAVE_PATH, 'w') as f:
                json.dump(trained_model, f, indent=4)
            print(f"   Modelo 'Most Popular' (json) guardado en: {MODEL_SAVE_PATH}")
        
        elif args.model_name == 'random_model':
            # No hay modelo que guardar
            print("   (El modelo 'Random' no tiene un objeto de modelo para guardar)")
        
        else:
            print(f"   Advertencia: No se ha definido una lógica de guardado para {args.model_name}")

    except Exception as e:
        print(f"   [ERROR] No se pudo guardar el modelo en {MODEL_SAVE_PATH}")
        print(f"   Detalle: {e}")
    # --- Fin de la nueva sección ---

    # --- 5. Medición de la Fase de PREDICCÍÓN ---
    print("\n--- MEASURING PREDICTION PHASE ---")
    prediction_tracker = EmissionsTracker(log_level='error')
    prediction_tracker.start()
    predictions_df = model_module.generate_predictions(trained_model, *prediction_extra_args)
    prediction_emissions = prediction_tracker.stop()
    print("--- PREDICTION MEASUREMENT FINISHED ---")

    # --- 6. Guardar Predicciones ---
    predictions_output_path = os.path.join(DATA_PATH, f"{args.model_name}_predictions.csv")
    predictions_df.to_csv(predictions_output_path, index=False)
    print(f"\nPredictions saved to '{predictions_output_path}'")

    # --- 7. Evaluación de Métricas ---
    print("\n--- EVALUATING METRICS ---")
    test_file = os.path.join(DATA_PATH, 'test.csv')
    test_df = pd.read_csv(test_file)
    
    # RMSE necesita el merge para comparar ratings reales vs. predichos.
    merged_df_for_rmse = pd.merge(test_df, predictions_df, on=['userId', 'movieId'], how='inner')
    rmse = calculate_rmse(merged_df_for_rmse)
    
    # Las métricas de ranking usan los dataframes completos.
    precision, recall, ndcg = calculate_ranking_metrics(predictions_df, test_df, k=10)
    
    print("--- METRICS CALCULATION FINISHED ---")

    # --- 8. Recopilar y Guardar Resultados en JSON ---
    final_results = {
        "training_footprint": {
            "co2_emissions_g": training_emissions * 1000,
            "energy_consumed_kWh": training_tracker.final_emissions_data.energy_consumed,
            "duration_seconds": training_tracker.final_emissions_data.duration
        },
        "prediction_footprint": {
            "co2_emissions_g": prediction_emissions * 1000,
            "energy_consumed_kWh": prediction_tracker.final_emissions_data.energy_consumed,
            "duration_seconds": prediction_tracker.final_emissions_data.duration
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
    print(f"  - RMSE              : {rmse:.4f}")
    print(f"  - Precision@10      : {precision:.4f}")
    print(f"  - Recall@10         : {recall:.4f}")
    print(f"  - nDCG@10           : {ndcg:.4f}")
    print(f"  - Training CO₂ (g)  : {final_results['training_footprint']['co2_emissions_g']:.4f}")
    print(f"  - Prediction CO₂ (g): {final_results['prediction_footprint']['co2_emissions_g']:.4f}")
    