import pandas as pd
import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuración ---
INPUT_FILE = 'individual_results.json'
# <<<<<<< CAMBIO AQUÍ >>>>>>>
OUTPUT_DIR = 'graficos/individuales' # Directorio donde se guardarán los gráficos
# --- Fin del Cambio ---

# <<<<<<< CONFIGURACIÓN DE MÉTRICA >>>>>>>
# Tu solicitud fue usar 'energy_kwh'.
# ADVERTENCIA ANALÍTICA: Como discutimos, 'energy_kwh' está subestimado en tu Mac (MPS).
# Para un análisis más robusto del costo computacional, cambia esto a 'duration_s'.
METRIC_TO_PLOT = 'energy_kwh' # Opciones: 'energy_kwh' o 'duration_s'
EXPECTED_DATASETS = [10, 25, 50, 75, 100]
# --------------------------------------

def plot_individual_results():
    """
    Carga 'individual_results.json', calcula el costo relativo de inferencia
    (basado en METRIC_TO_PLOT) y guarda un gráfico .png por cada modelo.
    """
    
    # --- 1. Cargar y Procesar los Datos ---
    print(f"--- Iniciando Análisis Gráfico de Inferencia Individual ---")
    print(f"Usando la métrica de costo: {METRIC_TO_PLOT}")
    
    if METRIC_TO_PLOT == 'energy_kwh':
        metric_title = "Energía Consumida (kWh)"
        Y_LABEL = metric_title
        METRIC_KEY = 'energy_kwh'
    else:
        metric_title = "Duración de Inferencia (s)"
        Y_LABEL = metric_title
        METRIC_KEY = 'duration_s'

    
    print(f"Cargando resultados desde {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Archivo no encontrado: {INPUT_FILE}", file=sys.stderr)
        print("Por favor, ejecuta 'get_top10_recommendations.py' (a través del script de notebook o .sh) primero.", file=sys.stderr)
        return

    with open(INPUT_FILE, 'r') as f:
        all_results = json.load(f)

    # Convertir el JSON anidado a un DataFrame plano
    data_to_analyze = []
    for dataset_perc, models_data in all_results.items():
        for model_name, users_data in models_data.items():
            for user_key, user_results in users_data.items():

                user_label = user_key
                try:
                    user_id = user_key.split(' (')[0].strip()
                    user_category = user_key.split(' (')[1].replace(')', '').strip()
                except IndexError:
                    user_id = user_key
                    user_category = 'Unknown'
                
                footprint = user_results.get('inference_footprint', {})
                duration_s = footprint.get('duration_seconds')
                energy_kwh = footprint.get('energy_consumed_kWh')
                
                # Seleccionar el valor de la métrica que usaremos
                metric_value = duration_s if METRIC_KEY == 'duration_s' else energy_kwh

                if metric_value is not None and metric_value > 0:
                    data_to_analyze.append({
                        'dataset': int(dataset_perc),
                        'model': model_name,
                        'user_id': user_id,
                        'user_category': user_category,
                        'user_label': user_label,
                        METRIC_KEY: metric_value
                    })

    if not data_to_analyze:
        print("[ERROR] No se encontraron datos de inferencia válidos en el archivo JSON.")
        return

    df = pd.DataFrame(data_to_analyze)
    print(f"Procesados {len(df)} registros de inferencia individual.")
    
    # Asegurarse de que el directorio de salida exista
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 2. Preparar datos y generar gráficos ---

    models_to_plot = df['model'].unique()
    print(f"Generando gráficos para {len(models_to_plot)} modelos...")

    df_plot = df.dropna(subset=[METRIC_KEY]).copy()
    df_plot = df_plot[df_plot[METRIC_KEY] > 0]
    df_plot = df_plot[df_plot['dataset'].isin(EXPECTED_DATASETS)]

    if df_plot.empty:
        print("[ERROR] No se encontraron valores positivos para la métrica seleccionada.")
        return

    missing_datasets = sorted(set(EXPECTED_DATASETS) - set(df_plot['dataset'].unique()))
    if missing_datasets:
        print(f"[Advertencia] No hay datos disponibles para los porcentajes: {missing_datasets}.")

    # --- 3. Guardar Gráficos Individualmente ---
    
    # Configurar estilo de Seaborn
    sns.set_theme(style="whitegrid")
    
    for model_name in models_to_plot:
        model_data = df_plot[df_plot['model'] == model_name].copy()
        
        if model_data.empty:
            print(f"  Omitiendo gráfico para {model_name} (sin datos válidos).")
            continue
            
        # Crear una nueva figura para este modelo
        plt.figure(figsize=(10, 6))
        
        # Crear un gráfico de líneas manual para evitar advertencias de pandas future
        unique_labels = model_data['user_label'].unique()
        palette = sns.color_palette('tab10', n_colors=len(unique_labels))
        label_to_color = {label: palette[idx % len(palette)] for idx, label in enumerate(unique_labels)}

        ax = plt.gca()
        for label in unique_labels:
            user_series = (
                model_data[model_data['user_label'] == label]
                .set_index('dataset')
                .reindex(EXPECTED_DATASETS)
            )

            if user_series[METRIC_KEY].notna().sum() == 0:
                continue

            ax.plot(
                EXPECTED_DATASETS,
                user_series[METRIC_KEY].values,
                marker='o',
                label=label,
                color=label_to_color[label]
            )
        
        ax.set_title(f"{metric_title} por Usuario: {model_name}", fontsize=16)
        ax.set_xlabel("Tamaño del Dataset de Entrenamiento (%)", fontsize=12)
        ax.set_ylabel(Y_LABEL, fontsize=12)
        ax.set_xticks(EXPECTED_DATASETS)
        ax.set_xlim(EXPECTED_DATASETS[0], EXPECTED_DATASETS[-1])
        ax.legend(title='Usuario', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Guardar la figura
        output_path = os.path.join(OUTPUT_DIR, f"costo_inferencia_{model_name}.png")
        try:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            print(f"  Gráfico guardado en: {output_path}")
        except Exception as e:
            print(f"  [ERROR] No se pudo guardar el gráfico {output_path}: {e}")
        
        plt.close() # Cerrar la figura para liberar memoria

    print("\n--- Todos los gráficos han sido generados. ---")

if __name__ == "__main__":
    plot_individual_results() # <<<<<<< CAMBIO AQUÍ: Llamar a la función

