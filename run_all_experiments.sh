#!/bin/bash
# Script para ejecutar todos los experimentos de modelos.

# Detener el script si un comando falla
set -e

# Lista de todos los modelos a ejecutar
MODELS=(
    "svd_model"
    "item_knn_model"
    "user_knn_model"
    "most_popular_model"
    "random_model"
    "lightfm_model"
    "ncf_model"
    "lightgcn_model"
    "multivae_model"
    "als_model" # <-- NUEVO MODELO AÑADIDO
)

# Lista de todos los porcentajes de dataset
PERCENTAGES=(
    "10"
    "25"
    "50"
    "75"
    "100"
)

# --- INICIO DE LA EJECUCIÓN ---
echo "=================================================================="
echo "INICIANDO EJECUCIÓN COMPLETA DE EXPERIMENTOS"
echo "Usando el script: run_experiment_saves.py"
echo "=================================================================="

# Bucle anidado para ejecutar cada modelo con cada dataset
for perc in "${PERCENTAGES[@]}"; do
    for model in "${MODELS[@]}"; do
        
        echo ""
        echo "------------------------------------------------------------------"
        echo "Ejecutando: Modelo: $model | Dataset: $perc%"
        echo "------------------------------------------------------------------"
        
        # Ejecutar el script principal de Python
        python3 run_experiment_saves.py --model_name "$model" --dataset_percentage "$perc"
        
        echo "Completado: $model | $perc%"
    
    done
done

echo "=================================================================="
echo "TODOS LOS EXPERIMENTOS HAN FINALIZADO."
echo "=================================================================="
