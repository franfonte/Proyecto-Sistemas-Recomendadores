import os
# Ya no se necesita ninguna configuración especial para codecarbon.

import argparse
import sys
import importlib
import json
import pandas as pd
from codecarbon import EmissionsTracker
from evaluate_results import calculate_rmse, calculate_ranking_metrics

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
    # <<<<<<< CAMBIO AQUÍ >>>>>>>
    # Se crea el tracker sin argumentos especiales para que use el modo más preciso por defecto.
    training_tracker = EmissionsTracker(log_level='error')
    training_tracker.start()
    trained_model = model_module.train_model(training_data)
    training_emissions = training_tracker.stop()
    print("--- TRAINING MEASUREMENT FINISHED ---")

    # --- 5. Medición de la Fase de PREDICCIÓN ---
    print("\n--- MEASURING PREDICTION PHASE ---")
    # <<<<<<< CAMBIO AQUÍ >>>>>>>
    # Se aplica la misma simplificación al segundo tracker.
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

