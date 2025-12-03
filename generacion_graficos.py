import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # para el gráfico 3D
import matplotlib as mpl
import numpy as np
import sys

# --- CONFIGURACIÓN GENERAL ---
mpl.rcParams['font.family'] = 'DejaVu Sans'  # evita el error con CO₂
sns.set(style="whitegrid", font_scale=1.2)
os.makedirs("graficos", exist_ok=True)
os.makedirs("graficos/general", exist_ok=True)
os.makedirs("graficos/familia", exist_ok=True)
os.makedirs("graficos/individuales", exist_ok=True)

# === CARGA DE DATOS ===
with open("results_updated.json", "r") as f:
    data = json.load(f)

# === TRANSFORMACIÓN A DATAFRAME ===
rows = []
for dataset_size, models in data.items():
    for model_name, values in models.items():
        perf = values["performance_metrics"]
        pred = values["prediction_footprint"]
        train = values["training_footprint"]
        rows.append({
            "dataset_size": int(dataset_size),
            "model": model_name,
            "ndcg_at_10": float(perf["ndcg_at_10"]),
            "map_at_10": float(perf["map_at_10"]),
            "prediction_energy_kWh": float(pred["energy_consumed_kWh"]),
            "training_energy_kWh": float(train["energy_consumed_kWh"]),
            "total_energy_kWh": float(pred["energy_consumed_kWh"] + train["energy_consumed_kWh"])
        })

df = pd.DataFrame(rows)

# === AGRUPACIÓN POR CATEGORÍA DE MODELO ===
grupo_map = {
    "random_model": "Línea Base",
    "most_popular_model": "Línea Base",
    "item_knn_model": "Filtrado Colaborativo Clásico",
    "user_knn_model": "Filtrado Colaborativo Clásico",
    "svd_model": "Factorización de Matrices Clásica",
    "als_model": "Factorización de Matrices Clásica",
    "lightfm_model": "Factorización de Matrices Clásica",
    "ncf_model": "Modelos de Redes Neuronales",
    "multivae_model": "Modelos de Redes Neuronales",
    "lightgcn_model": "Modelos de Redes Neuronales"
}
df["grupo"] = df["model"].map(grupo_map)
df["model_display"] = df["model"].str.replace("_model", "", regex=False)
df = df.dropna(subset=["grupo"]).reset_index(drop=True)
dataset_ticks = sorted(df["dataset_size"].unique())
dataset_labels = [f"{int(x)}%" for x in dataset_ticks]

# === PALETAS DE COLOR ===
paletas = {
    "Línea Base": sns.color_palette("Blues", n_colors=2),
    "Filtrado Colaborativo Clásico": sns.color_palette("Greens", n_colors=2),
    "Factorización de Matrices Clásica": sns.color_palette("Oranges", n_colors=3),
    "Modelos de Redes Neuronales": sns.color_palette("Purples", n_colors=3)
}

# --- FUNCIÓN PARA GRAFICAR COMPARATIVOS ---
def plot_line(metric, ylabel, filename, logy=False):
    plt.figure(figsize=(10,6))

    for grupo, subdf in df.groupby("grupo"):
        modelos = subdf["model"].unique()
        colores = paletas[grupo]
        for i, modelo in enumerate(modelos):
            datos = subdf[subdf["model"] == modelo].sort_values("dataset_size")
            display_name = datos["model_display"].iloc[0]
            plt.plot(
                datos["dataset_size"],
                datos[metric],
                label=f"{display_name} ({grupo})",
                color=colores[i % len(colores)],
                linestyle="-",
                marker="o",
                linewidth=2.2
            )
            # Etiqueta final del modelo
            plt.text(
                datos["dataset_size"].max() * 1.01,
                datos[metric].iloc[-1],
                display_name,
                fontsize=9,
                color=colores[i % len(colores)],
                va="center"
            )

    plt.title(f"{ylabel} vs Tamaño del Dataset", fontsize=15)
    plt.xlabel("Tamaño del Dataset")
    plt.ylabel(ylabel)
    if logy:
        plt.yscale("log")
    plt.xticks(dataset_ticks, dataset_labels)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"graficos/general/{filename}.png", dpi=300)
    plt.close()


# === GRAFICO 1: NDCG vs Dataset ===
plot_line("ndcg_at_10", "NDCG@10", "ndcg_vs_dataset")

# === GRAFICO 1.5: MAP vs Dataset ===
plot_line("map_at_10", "MAP@10", "map_vs_dataset")

# === GRAFICO 2: kWh vs Dataset ===
plot_line("total_energy_kWh", "Energía Total (kWh)", "kwh_vs_dataset", logy=True)

# === GRAFICO 2b: kWh Entrenamiento vs Dataset ===
plot_line("training_energy_kWh", "Energía Entrenamiento (kWh)", "kwh_vs_dataset_training", logy=True)

# === GRAFICO 2c: kWh Predicción vs Dataset ===
plot_line("prediction_energy_kWh", "Energía Predicción (kWh)", "kwh_vs_dataset_prediction", logy=True)

# === GRAFICO 3: Trade-off kWh vs NDCG ===
plt.figure(figsize=(9,6))
for grupo, subdf in df.groupby("grupo"):
    modelos = subdf["model"].unique()
    colores = paletas[grupo]
    for i, modelo in enumerate(modelos):
        datos = subdf[subdf["model"] == modelo].sort_values("dataset_size")
        display_name = datos["model_display"].iloc[0]
        plt.plot(
            datos["total_energy_kWh"],
            datos["ndcg_at_10"],
            label=f"{display_name} ({grupo})",
            color=colores[i % len(colores)],
            linestyle="-",
            marker="o",
            linewidth=2.2
        )
        for energy, ndcg, size in zip(datos["total_energy_kWh"], datos["ndcg_at_10"], datos["dataset_size"]):
            plt.annotate(
                f"{int(size)}%",
                xy=(energy, ndcg),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color=colores[i % len(colores)]
            )
plt.xlabel("Energía Total (kWh)")
plt.ylabel("NDCG@10")
plt.title("Trade-off entre Precisión (NDCG@10) y Energía Consumida", fontsize=15)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("graficos/general/tradeoff_ndcg_vs_energy.png", dpi=300)
plt.close()

# === GRAFICO 3.5: Trade-off kWh vs MAP ===
plt.figure(figsize=(9,6))
for grupo, subdf in df.groupby("grupo"):
    modelos = subdf["model"].unique()
    colores = paletas[grupo]
    for i, modelo in enumerate(modelos):
        datos = subdf[subdf["model"] == modelo].sort_values("dataset_size")
        display_name = datos["model_display"].iloc[0]
        plt.plot(
            datos["total_energy_kWh"],
            datos["map_at_10"],
            label=f"{display_name} ({grupo})",
            color=colores[i % len(colores)],
            linestyle="-",
            marker="o",
            linewidth=2.2
        )
        for energy, map_val, size in zip(datos["total_energy_kWh"], datos["map_at_10"], datos["dataset_size"]):
            plt.annotate(
                f"{int(size)}%",
                xy=(energy, map_val),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color=colores[i % len(colores)]
            )
plt.xlabel("Energía Total (kWh)")
plt.ylabel("MAP@10")
plt.title("Trade-off entre Precisión (MAP@10) y Energía Consumida", fontsize=15)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("graficos/general/tradeoff_map_vs_energy.png", dpi=300)
plt.close()

# === GRAFICO 4: 3D con Matplotlib ===
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')

for grupo, subdf in df.groupby("grupo"):
    modelos = subdf["model"].unique()
    colores = paletas[grupo]
    for i, modelo in enumerate(modelos):
        datos = subdf[subdf["model"] == modelo].sort_values("dataset_size")
        display_name = datos["model_display"].iloc[0]
        ax.plot(
            datos["dataset_size"],
            datos["total_energy_kWh"],
            datos["ndcg_at_10"],
            color=colores[i % len(colores)],
            linestyle="-",
            marker="o",
            label=f"{display_name} ({grupo})"
        )
        for x_coord, y_coord, z_coord in zip(
            datos["dataset_size"], datos["total_energy_kWh"], datos["ndcg_at_10"]
        ):
            ax.plot(
                [x_coord, x_coord],
                [y_coord, y_coord],
                [0, z_coord],
                linestyle="--",
                linewidth=1.0,
                color=colores[i % len(colores)]
            )

ax.set_title("3D: Tamaño Dataset - Energía Total - NDCG@10", fontsize=13)
ax.set_xlabel("Tamaño Dataset")
ax.set_ylabel("Energía Total (kWh)", labelpad=12)
ax.set_zlabel("NDCG@10")
ax.set_xticks(dataset_ticks)
ax.set_xticklabels(dataset_labels)
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("graficos/general/3D_dataset_energy_ndcg.png", dpi=300)
plt.close()

print("✅ Gráficos generados correctamente:")
print(" - graficos/general/ndcg_vs_dataset.png")
print(" - graficos/general/map_vs_dataset.png")
print(" - graficos/general/kwh_vs_dataset.png")
print(" - graficos/general/kwh_vs_dataset_training.png")
print(" - graficos/general/kwh_vs_dataset_prediction.png")
print(" - graficos/general/tradeoff_ndcg_vs_energy.png")
print(" - graficos/general/tradeoff_map_vs_energy.png")
print(" - graficos/general/3D_dataset_energy_ndcg.png (3D con Matplotlib)")

# ==========================================
# === SECCIÓN: GRÁFICOS POR FAMILIA ===
# ==========================================

print("--- Generando gráficos promedio por familia ---")

# 1. Agrupación de datos (Promedio por familia y tamaño de dataset)
df_familia = df.groupby(["grupo", "dataset_size"]).agg({
    "ndcg_at_10": "mean",
    "map_at_10": "mean",
    "total_energy_kWh": "mean",
    "training_energy_kWh": "mean",
    "prediction_energy_kWh": "mean"
}).reset_index()

# 2. Función adaptada para familias (Mismo estilo)
def plot_line_familia(metric, ylabel, filename, logy=False):
    plt.figure(figsize=(10,6))
    
    # Iteramos por los grupos únicos
    for grupo in df_familia["grupo"].unique():
        datos = df_familia[df_familia["grupo"] == grupo].sort_values("dataset_size")
        
        # Usamos el color más oscuro de la paleta original para el promedio
        color_linea = paletas[grupo][-1] 
        
        plt.plot(
            datos["dataset_size"],
            datos[metric],
            label=f"Promedio {grupo}",
            color=color_linea,
            linestyle="-",
            marker="D",  # Diamante para distinguir que es promedio
            linewidth=2.5
        )
        
        # Removed end-of-line text labels

    plt.title(f"{ylabel} vs Tamaño del Dataset (Promedio por Familia)", fontsize=15)
    plt.xlabel("Tamaño del Dataset")
    plt.ylabel(ylabel)
    if logy:
        plt.yscale("log")
    plt.xticks(dataset_ticks, dataset_labels)
    # Leyenda dentro del gráfico, esquina superior izquierda
    plt.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=9, frameon=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"graficos/familia/{filename}_familia.png", dpi=300)
    plt.close()

# === GENERACIÓN DE LOS GRÁFICOS DE LÍNEA (FAMILIA) ===
plot_line_familia("ndcg_at_10", "NDCG@10 Promedio", "ndcg_vs_dataset")
plot_line_familia("map_at_10", "MAP@10 Promedio", "map_vs_dataset")
plot_line_familia("total_energy_kWh", "Energía Total Promedio (kWh)", "kwh_vs_dataset", logy=True)
plot_line_familia("training_energy_kWh", "Energía Entrenamiento Promedio (kWh)", "kwh_vs_dataset_training", logy=True)
plot_line_familia("prediction_energy_kWh", "Energía Predicción Promedio (kWh)", "kwh_vs_dataset_prediction", logy=True)

# === GRÁFICO TRADE-OFF (FAMILIA) ===
plt.figure(figsize=(9,6))
for grupo in df_familia["grupo"].unique():
    datos = df_familia[df_familia["grupo"] == grupo].sort_values("dataset_size")
    color_linea = paletas[grupo][-1]
    
    plt.plot(
        datos["total_energy_kWh"],
        datos["ndcg_at_10"],
        label=f"Promedio {grupo}",
        color=color_linea,
        linestyle="-",
        marker="D",
        linewidth=2.5
    )
    
    for energy, ndcg, size in zip(datos["total_energy_kWh"], datos["ndcg_at_10"], datos["dataset_size"]):
        plt.annotate(
            f"{int(size)}%",
            xy=(energy, ndcg),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            color=color_linea
        )

plt.xlabel("Energía Total Promedio (kWh)")
plt.ylabel("NDCG@10 Promedio")
plt.title("Trade-off Precisión vs Energía (Promedio por Familia)", fontsize=15)
# Leyenda dentro del gráfico, esquina superior izquierda
plt.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=9, frameon=True, fancybox=True, framealpha=0.9)
plt.tight_layout()
plt.savefig("graficos/familia/tradeoff_ndcg_vs_energy_familia.png", dpi=300)
plt.close()

# === GRÁFICO TRADE-OFF (FAMILIA) map ===
plt.figure(figsize=(9,6))
for grupo in df_familia["grupo"].unique():
    datos = df_familia[df_familia["grupo"] == grupo].sort_values("dataset_size")
    color_linea = paletas[grupo][-1]
    
    plt.plot(
        datos["total_energy_kWh"],
        datos["map_at_10"],
        label=f"Promedio {grupo}",
        color=color_linea,
        linestyle="-",
        marker="D",
        linewidth=2.5
    )
    
    for energy, map_val, size in zip(datos["total_energy_kWh"], datos["map_at_10"], datos["dataset_size"]):
        plt.annotate(
            f"{int(size)}%",
            xy=(energy, map_val),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            color=color_linea
        )

plt.xlabel("Energía Total Promedio (kWh)")
plt.ylabel("MAP@10 Promedio")
plt.title("Trade-off Precisión vs Energía (Promedio por Familia)", fontsize=15)
# Leyenda dentro del gráfico, esquina superior izquierda
plt.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=9, frameon=True, fancybox=True, framealpha=0.9)
plt.tight_layout()
plt.savefig("graficos/familia/tradeoff_map_vs_energy_familia.png", dpi=300)
plt.close()

# === GRÁFICO 3D (FAMILIA) ===
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')

for grupo in df_familia["grupo"].unique():
    datos = df_familia[df_familia["grupo"] == grupo].sort_values("dataset_size")
    color_linea = paletas[grupo][-1]
    
    ax.plot(
        datos["dataset_size"],
        datos["total_energy_kWh"],
        datos["ndcg_at_10"],
        color=color_linea,
        linestyle="-",
        marker="D",
        label=f"Promedio {grupo}",
        linewidth=2
    )
    
    # Líneas de proyección vertical
    for x_coord, y_coord, z_coord in zip(
        datos["dataset_size"], datos["total_energy_kWh"], datos["ndcg_at_10"]
    ):
        ax.plot(
            [x_coord, x_coord],
            [y_coord, y_coord],
            [0, z_coord],
            linestyle="--",
            linewidth=1.0,
            color=color_linea,
            alpha=0.5
        )

ax.set_title("3D: Dataset - Energía - NDCG (Promedio por Familia)", fontsize=13)
ax.set_xlabel("Tamaño Dataset")
ax.set_ylabel("Energía Total (kWh)", labelpad=12)
ax.set_zlabel("NDCG@10")
ax.set_xticks(dataset_ticks)
ax.set_xticklabels(dataset_labels)
# Leyenda dentro del gráfico 3D, esquina superior izquierda
ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=8, frameon=True, fancybox=True, framealpha=0.9)
plt.tight_layout()
plt.savefig("graficos/familia/3D_dataset_energy_ndcg_familia.png", dpi=300)
plt.close()

print("✅ Gráficos por familia generados correctamente con sufijo '_familia'.")

# =============================================
# === SECCIÓN: GRÁFICOS POR TIPO DE USUARIO ===
# =============================================

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

plot_individual_results()