#!/bin/bash
#
# Este script prueba la generación de Top-10 individual para una LISTA
# de usuarios y una LISTA de datasets, pasando la categoría del usuario.

# --- CONFIGURACIÓN ---
# <<<<<<< CAMBIO AQUÍ: Formato "ID:Categoría" >>>>>>>
# Añadir usuarios y sus categorías aquí, separados por dos puntos.
USER_DATA=(
    "4169:Power User"
    "4226:Cold-Start User"
    "1941:Critic"
    "4277:Fan"
    "4169:Niche Seeker"
    "5557:Mainstream User"
    # Añade más "ID:Categoría" aquí si es necesario
)

# Pon los porcentajes de dataset que quieras probar
DATASET_PERCS=("10" "25" "50" "75" "100") # Ejemplo: ("10" "25" "50" "75" "100")

# Lista de modelos a probar (todos los que son compatibles)
MODELS=(
    "svd_model"
    "item_knn_model"
    "user_knn_model"
    "lightfm_model"
    "als_model"
    "ncf_model"
    "multivae_model"
    "most_popular_model"
    "random_model"
)
# --- FIN CONFIGURACIÓN ---


echo "--- Iniciando prueba de predicción individual ---"
echo "Usando script: get_top10_recommendations.py"
echo "=============================================================="

# Bucle anidado para probar cada combinación
for perc in "${DATASET_PERCS[@]}"; do
    # <<<<<<< CAMBIO AQUÍ: Bucle sobre los datos de usuario >>>>>>>
    for user_entry in "${USER_DATA[@]}"; do
        
        # <<<<<<< CAMBIO AQUÍ: Parsear ID y Categoría >>>>>>>
        # IFS=: read -r user_id user_category <<< "$user_entry" # Método alternativo
        user_id=$(echo "$user_entry" | cut -d':' -f1)
        user_category=$(echo "$user_entry" | cut -d':' -f2)

        echo ""
        echo "**************************************************************"
        echo "PROBANDO: Dataset: $perc% | Usuario: $user_id ($user_category)"
        echo "**************************************************************"
        
        # Bucle interno para cada modelo
        for model in "${MODELS[@]}"; do
            
            echo "\n--- Modelo: $model ---"
            
            # <<<<<<< CAMBIO AQUÍ: Añadir argumento --user_category >>>>>>>
            python3 get_top10_recommendations.py \
                --model_name "$model" \
                --dataset_percentage "$perc" \
                --user_id "$user_id" \
                --user_category "$user_category"
            
        done
        
        echo "\n--- Omitiendo: lightgcn_model (no soportado por el script de predicción individual) ---"

    done
done

echo "=============================================================="
echo "--- Pruebas de predicción individual finalizadas ---"
