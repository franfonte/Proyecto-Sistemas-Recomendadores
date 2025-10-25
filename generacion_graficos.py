import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # para el gráfico 3D
import matplotlib as mpl

# --- CONFIGURACIÓN GENERAL ---
mpl.rcParams['font.family'] = 'DejaVu Sans'  # evita el error con CO₂
sns.set(style="whitegrid", font_scale=1.2)
os.makedirs("graficos", exist_ok=True)

# === CARGA DE DATOS ===
with open("results.json", "r") as f:
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
    plt.savefig(f"graficos/{filename}.png", dpi=300)
    plt.close()


# === GRAFICO 1: NDCG vs Dataset ===
plot_line("ndcg_at_10", "NDCG@10", "ndcg_vs_dataset")

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
plt.savefig("graficos/tradeoff_ndcg_vs_energy.png", dpi=300)
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
plt.savefig("graficos/3D_dataset_energy_ndcg.png", dpi=300)
plt.close()

print("✅ Gráficos generados correctamente:")
print(" - graficos/ndcg_vs_dataset.png")
print(" - graficos/kwh_vs_dataset.png")
print(" - graficos/kwh_vs_dataset_training.png")
print(" - graficos/kwh_vs_dataset_prediction.png")
print(" - graficos/tradeoff_ndcg_vs_energy.png")
print(" - graficos/3D_dataset_energy_ndcg.png (3D con Matplotlib)")