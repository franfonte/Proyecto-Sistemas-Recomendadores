import pandas as pd
import numpy as np
import json
import os
import sys

# --- Configuración ---
INPUT_FILE = 'individual_results.json'
OUTPUT_DIR = 'analisis'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'descubrimientos_inferencia_individual.txt')

def analyze_individual_results():
    """
    Carga 'individual_results.json', realiza un análisis numérico
    y guarda los hallazgos en un archivo .txt.
    """
    
    # --- 1. Cargar y Procesar los Datos ---
    print(f"Cargando resultados desde {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Archivo no encontrado: {INPUT_FILE}", file=sys.stderr)
        print("Por favor, ejecuta 'get_top10_recommendations.py' (a través del script de notebook o .sh) primero.", file=sys.stderr)
        return

    with open(INPUT_FILE, 'r') as f:
        all_results = json.load(f)

    # Convertir el JSON anidado a un DataFrame plano (lista de diccionarios)
    data_to_analyze = []

    for dataset_perc, models_data in all_results.items():
        for model_name, users_data in models_data.items():
            for user_key, user_results in users_data.items():
                
                # Extraer ID y Categoría de la clave
                try:
                    user_id = user_key.split(' (')[0]
                    user_category = user_key.split(' (')[1].replace(')', '')
                except IndexError:
                    user_id = user_key
                    user_category = 'Unknown'
                
                footprint = user_results.get('inference_footprint', {})
                duration_s = footprint.get('duration_seconds')
                energy_kwh = footprint.get('energy_consumed_kWh')
                co2_g = footprint.get('co2_emissions_g')

                if duration_s is not None and energy_kwh is not None:
                    data_to_analyze.append({
                        'dataset': int(dataset_perc),
                        'model': model_name,
                        'user_id': user_id,
                        'user_category': user_category,
                        'duration_s': duration_s,
                        'energy_kwh': energy_kwh,
                        'co2_g': co2_g
                    })

    if not data_to_analyze:
        print("[ERROR] No se encontraron datos de inferencia válidos en el archivo JSON.")
        return

    df = pd.DataFrame(data_to_analyze)
    print(f"Procesados {len(df)} registros de inferencia individual.")
    
    # Asegurarse de que el directorio de salida exista
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 2. Realizar Análisis Numérico ---
    
    # Iniciar el string del reporte
    report = f"Análisis Numérico de Resultados de Inferencia Individual (Top-10)\n"
    report += f"=================================================================\n"
    report += f"Generado el: {pd.Timestamp.now()}\n"
    report += f"Basado en {len(df)} mediciones de '{INPUT_FILE}'\n\n"

    report += "Definiciones de Perfiles de Usuario\n"
    report += "------------------------------------\n"
    report += "--- Power User ---\nDescripción: Usuarios que están consistentemente en el 20% superior de actividad. Se selecciona el más activo.\n\n"
    report += "--- Cold-Start User ---\nDescripción: Usuarios que están consistentemente en el 20% inferior de actividad. Se selecciona el menos activo.\n\n"
    report += "--- Critic ---\nDescripción: Usuarios que están consistentemente en el 20% inferior de rating promedio. Se selecciona el más activo de este grupo.\n\n"
    report += "--- Fan ---\nDescripción: Usuarios que están consistentemente en el 20% superior de rating promedio. Se selecciona el más activo de este grupo.\n\n"
    report += "--- Niche Seeker ---\nDescripción: Usuarios que consistentemente prefieren ítems de baja popularidad (long-tail). Se selecciona el más activo de este grupo.\n\n"
    report += "--- Mainstream User ---\nDescripción: Usuarios que consistentemente prefieren ítems de alta popularidad (blockbusters). Se selecciona el más activo de este grupo.\n\n"

    # --- Análisis 1: Costo Promedio por Modelo (en todos los datasets y usuarios) ---
    report += "--- Análisis 1: Costo Promedio por Modelo (General) ---\n"
    avg_cost_by_model = (
        df.groupby('model')[['duration_s', 'energy_kwh']]
        .agg(['mean', 'std', 'min', 'max'])
        .sort_values(('duration_s', 'mean'), ascending=False)
    )
    report += "Duración (s) y energía (kWh): media, desviación estándar, mínimo y máximo. Ordenado por mayor duración media.\n"
    report += avg_cost_by_model.to_string(float_format="%.8f")
    report += "\n\n"

    duration_means = avg_cost_by_model[('duration_s', 'mean')]
    energy_means = avg_cost_by_model[('energy_kwh', 'mean')]
    report += f"Duración media más alta: {duration_means.idxmax()} = {duration_means.max():.4f}s.\n"
    report += f"Duración media más baja: {duration_means.idxmin()} = {duration_means.min():.4f}s.\n"
    report += f"Energía media más alta: {energy_means.idxmax()} = {energy_means.max():.8f} kWh.\n"
    report += f"Energía media más baja: {energy_means.idxmin()} = {energy_means.min():.8f} kWh.\n"
    report += "\n\n"

    # --- Análisis 2: Costo Promedio por Perfil de Usuario (en todos los datasets y modelos) ---
    report += "--- Análisis 2: Costo Promedio por Perfil de Usuario (General) ---\n"
    avg_cost_by_user = (
        df.groupby('user_category')[['duration_s', 'energy_kwh']]
        .agg(['mean', 'std', 'min', 'max'])
        .sort_values(('duration_s', 'mean'), ascending=False)
    )
    report += "Duración (s) y energía (kWh): media, desviación estándar, mínimo y máximo por perfil de usuario.\n"
    report += avg_cost_by_user.to_string(float_format="%.8f")
    report += "\n\n"

    if ('Power User' in avg_cost_by_user.index) and ('Cold-Start User' in avg_cost_by_user.index):
        power_duration = avg_cost_by_user.loc['Power User', ('duration_s', 'mean')]
        cold_duration = avg_cost_by_user.loc['Cold-Start User', ('duration_s', 'mean')]
        if cold_duration > 0:
            ratio = power_duration / cold_duration
            report += f"Relación de duración media Power User / Cold-Start User: {ratio:.2f}x.\n"
    report += "\n\n"
    
    # --- Análisis 3: Resumen por tamaño de dataset ---
    report += "--- Análisis 3: Resumen Estadístico por Tamaño de Dataset ---\n"
    dataset_summary = (
        df.groupby('dataset')[['duration_s', 'energy_kwh']]
        .agg(['mean', 'std', 'min', 'max'])
        .sort_index()
    )
    report += "Duración (s) y energía (kWh) media, desviación estándar, mínimo y máximo por porcentaje del dataset.\n"
    report += dataset_summary.to_string(float_format="%.8f")
    report += "\n\n"

    # --- Análisis 4: Escalabilidad (Costo de 10% vs 100%) ---
    report += "--- Análisis 4: Escalabilidad de Inferencia (10% vs 100% del dataset) ---\n"
    report += "Comparativa de duración y energía medias entre entrenamientos con 10% y 100% del dataset.\n"
    
    try:
        cost_10 = df[df['dataset'] == 10].groupby('model')[['duration_s', 'energy_kwh']].mean()
        cost_100 = df[df['dataset'] == 100].groupby('model')[['duration_s', 'energy_kwh']].mean()
        
        scalability_df = pd.DataFrame({
            'duration_10': cost_10['duration_s'],
            'duration_100': cost_100['duration_s'],
            'energy_10_kwh': cost_10['energy_kwh'],
            'energy_100_kwh': cost_100['energy_kwh']
        }).dropna()

        scalability_df['ratio_duracion_100_vs_10'] = scalability_df['duration_100'] / scalability_df['duration_10']
        scalability_df['ratio_energia_100_vs_10'] = scalability_df['energy_100_kwh'] / scalability_df['energy_10_kwh']

        report += "Duración 10% vs 100% (media por modelo):\n"
        report += (
            scalability_df[['duration_10', 'duration_100', 'ratio_duracion_100_vs_10']]
            .sort_values('ratio_duracion_100_vs_10', ascending=False)
            .to_string(float_format="%.4f")
        )
        report += "\n\n"

        report += "Energía 10% vs 100% (media por modelo):\n"
        report += (
            scalability_df[['energy_10_kwh', 'energy_100_kwh', 'ratio_energia_100_vs_10']]
            .sort_values('ratio_energia_100_vs_10', ascending=False)
            .to_string(float_format="%.8f")
        )
        report += "\n\n"

        max_ratio_duration = scalability_df['ratio_duracion_100_vs_10'].idxmax()
        min_ratio_duration = scalability_df['ratio_duracion_100_vs_10'].idxmin()
        report += (
            f"Mayor relación duración 100%/10%: {max_ratio_duration} = "
            f"{scalability_df.loc[max_ratio_duration, 'ratio_duracion_100_vs_10']:.2f}x.\n"
        )
        report += (
            f"Menor relación duración 100%/10%: {min_ratio_duration} = "
            f"{scalability_df.loc[min_ratio_duration, 'ratio_duracion_100_vs_10']:.2f}x.\n"
        )
    
    except KeyError:
        report += "[No se pudo calcular la escalabilidad, faltan datos del 10% o 100%]\n"
    except Exception as e:
        report += f"[Error al calcular escalabilidad: {e}]\n"
        
    # --- Análisis 5: Rangos Globales ---
    report += "\n\n--- Análisis 5: Valores Extremos Registrados ---\n"
    global_stats = df[['duration_s', 'energy_kwh']].agg(['min', 'max', 'mean', 'median'])
    report += "Duración (s) y energía (kWh) mínima, máxima, media y mediana considerando todas las mediciones.\n"
    report += global_stats.to_string(float_format="%.8f")
    report += "\n\n"

    # --- Análisis 6: Desglose por Perfil de Usuario y Modelo ---
    report += "--- Análisis 6: Desglose por Perfil de Usuario y Modelo ---\n"
    user_model_summary = (
        df.groupby(['user_category', 'model'])[['duration_s', 'energy_kwh']]
        .agg(['mean', 'std', 'min', 'max'])
        .sort_index()
    )
    report += "Duración (s) y energía (kWh) media, desviación estándar, mínimo y máximo para cada combinación perfil-modelo.\n"
    report += user_model_summary.to_string(float_format="%.8f")
    report += "\n\n"

    for user_category, subset in df.groupby('user_category'):
        model_means = subset.groupby('model')[['duration_s', 'energy_kwh']].mean()
        duration_max_model = model_means['duration_s'].idxmax()
        duration_min_model = model_means['duration_s'].idxmin()
        energy_max_model = model_means['energy_kwh'].idxmax()
        energy_min_model = model_means['energy_kwh'].idxmin()
        report += (
            f"Perfil '{user_category}': duración media más alta en {duration_max_model} = "
            f"{model_means.loc[duration_max_model, 'duration_s']:.4f}s; "
            f"duración media más baja en {duration_min_model} = "
            f"{model_means.loc[duration_min_model, 'duration_s']:.4f}s.\n"
        )
        report += (
            f"Perfil '{user_category}': energía media más alta en {energy_max_model} = "
            f"{model_means.loc[energy_max_model, 'energy_kwh']:.8f} kWh; "
            f"energía media más baja en {energy_min_model} = "
            f"{model_means.loc[energy_min_model, 'energy_kwh']:.8f} kWh.\n"
        )
    report += "\n"

    report += "--- FIN DEL REPORTE ---"
    
    # --- 3. Guardar el Reporte ---
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n¡Análisis completado! Los descubrimientos se han guardado en:")
        print(f"{OUTPUT_FILE}")
    except Exception as e:
        print(f"\n[ERROR] No se pudo escribir el archivo de reporte en {OUTPUT_FILE}: {e}")

if __name__ == "__main__":
    analyze_individual_results()