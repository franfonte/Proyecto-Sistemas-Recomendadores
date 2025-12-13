# Proyecto Sistemas de Recomendación con Medición de Huella de Carbono

Este repositorio orquesta experimentos de recomendación sobre MovieLens 1M y mide su impacto ambiental con CodeCarbon. Incluye pipelines para preparar datos, entrenar modelos heterogéneos, conservar huellas de entrenamiento e inferencia, analizar resultados y generar visualizaciones.

## Características Clave

- Seguimiento detallado de CO₂, energía y duración en entrenamiento e inferencia, alimentado por CodeCarbon.
- Catálogo de 10 modelos (filtrado clásico, factorización, redes neuronales y baselines) definidos en `models/`.
- Automatización para ejecutar todos los modelos en subconjuntos de 10% a 100% con `run_all_experiments.sh`.
- Evaluaciones que calculan métricas de ranking (Precision@10, Recall@10, nDCG@10, MAP@10, Hit-Rate, MRR) y diversidad/novelty.
- Herramientas para auditar recomendaciones individuales, filtrar ítems ya vistos y medir el costo energético de cada listado Top‑10.
- Scripts de análisis numérico y generación de gráficos comparativos guardados en `analisis/` y `graficos/`.

## Panorama del Repositorio

```
.
├── README.md
├── requirements.txt                # Dependencias principales
├── preprocessdata.py               # Conversión MovieLens .dat ➜ .csv
├── prepare_datasets.py             # Submuestreo estratificado y splits 80/20 + antitest
├── run_experiment_saves.py         # Ejecución estándar con emisiones + métricas + guardado de modelos
├── run_all_experiments.sh          # Lanza run_experiment_saves.py para todos los modelos y datasets
├── test_all_individual_predictions.sh # Ejecuta get_top10_recommendations.py para varios usuarios tipo
├── evaluate_results.py             # Cálculo de RMSE y métricas@10 desde predicciones
├── evaluate_new_results.py         # Amplía results.json con hit-rate, MAP, MRR, novelty, diversity
├── get_top10_recommendations.py    # Predicciones Top‑10 para un usuario, con huella energética
├── predict_single.py               # Verifica que una predicción individual coincida con el batch
├── create_readable.py              # Convierte IDs a títulos, genera ejemplos legibles
├── generacion_graficos.py          # Visualizaciones a partir de results_updated.json
├── analisis_numericos_resultados*.py # Reportes y tablas en analisis/
├── data/                           # MovieLens procesado + subconjuntos 10/25/50/75/100
├── models/                         # Implementaciones de algoritmos de recomendación
├── trained_models/                 # Modelos entrenados y mapas usuario/ítem por porcentaje
├── results.json                    # Métricas + huellas por modelo y dataset
├── results_updated.json            # results.json enriquecido con métricas adicionales
├── individual_results.json         # Huella de recomendaciones Top‑10 por usuario tipo
├── individual_results_examples.json # Versión legible (títulos MovieLens)
├── graficos/                       # Gráficos agregados (general, familia, individuales)
└── analisis/                       # Tablas y hallazgos exportados
```

## Requisitos Previos

- Python 3.10 o superior
- Dataset MovieLens 1M descargado desde [GroupLens](https://grouplens.org/datasets/movielens/1m/)
- Dependencias listadas en `requirements.txt`

## Configuración del Entorno

```bash
python3 -m venv venv
source venv/bin/activate            # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Preparación de Datos

1. Ubica el directorio original `ml-1m` dentro de `data/`.
2. Convierte los archivos `.dat` a `.csv` (una sola vez):
     ```bash
     python preprocessdata.py
     ```
3. Genera subconjuntos estratificados (10%, 25%, 50%, 75%, 100%) y sus splits `train/test/antitest`:
     ```bash
     python prepare_datasets.py
     ```

## Ejecución de Experimentos Masivos

`run_experiment_saves.py` es el punto de entrada principal. Para un modelo y porcentaje determinados:

```bash
python run_experiment_saves.py \
        --model_name lightfm_model \
        --dataset_percentage 50
```

El script realiza los pasos siguientes:

- Invoca `preprocess_data()` del modelo para obtener datos listos y mapas opcionales de IDs.
- Mide la fase de entrenamiento con CodeCarbon (CO₂, kWh, segundos) y guarda el modelo entrenado en `trained_models/<porcentaje>/`.
- Ejecuta `generate_predictions()` midiendo también la inferencia.
- Guarda las predicciones en `data/<porcentaje>/<modelo>_predictions.csv`.
- Calcula RMSE, Precision@10, Recall@10 y nDCG@10 (si hay solapamiento con `test.csv`).
- Actualiza `results.json` con métricas y huellas, serializando a JSON seguro para tipos NumPy.

### Ejecución para Todos los Modelos y Subconjuntos

```bash
bash run_all_experiments.sh
```

El guion recorre las listas `MODELS` y `PERCENTAGES`. Modifícalas para incluir/excluir modelos o porcentajes. Las fallas de un modelo no detienen el resto del experimento.

## Evaluación y Visualizaciones

- `evaluate_results.py` ya se emplea dentro de `run_experiment_saves.py` para métricas base (RMSE, Precision@10, Recall@10, nDCG@10).
- `evaluate_new_results.py` añade Hit-Rate@10, MAP@10, MRR@10, Novelty@10 (bits), y Diversity@10 (1 – similitud promedio) utilizando popularidad, matrices usuario‑ítem y similitud coseno. Ejecuta:
    ```bash
    python evaluate_new_results.py \
            --results results.json \
            --data-dir data \
            --output results_updated.json
    ```
- `generacion_graficos.py` transforma `results_updated.json` en gráficos (líneas, trade-offs, 3D) dentro de `graficos/general/` y promedios por familia en `graficos/familia/`.
- `analisis_numericos_resultados.py` y `analisis_numericos_resultados_individuales.py` exportan CSV y reportes `.txt` con porcentajes de cambio, eficiencia energética y análisis por perfil. Los archivos se guardan en `analisis/`.

## Predicciones Individuales y Auditoría

- `get_top10_recommendations.py` genera un Top‑10 por usuario:
    ```bash
    python get_top10_recommendations.py \
            --model_name ncf_model \
            --dataset_percentage 100 \
            --user_id 4169 \
            --user_category "Power User"
    ```
    Acciones clave:
    - Carga el modelo entrenado y sus mapas (`trained_models/<porcentaje>/`).
    - Predice puntuaciones para todos los ítems, filtra los ya vistos (según `train.csv`) y mide la huella de esa inferencia con CodeCarbon.
    - Agrega resultados a `individual_results.json` (organizado por porcentaje, modelo y perfil) con emisiones, energía y duración.

- `test_all_individual_predictions.sh` automatiza la ejecución anterior para una lista de perfiles (`USER_DATA`) y porcentajes (`DATASET_PERCS`). Ejecuta todos los modelos compatibles y omite `lightgcn_model`, que no soporta inferencia individual en este flujo.

- `predict_single.py` contrasta la predicción puntual (`user_id`, `movie_id`) con el batch almacenado en `data/<porcentaje>/<modelo>_predictions.csv`, útil para depuración.

- `create_readable.py` enriquece `individual_results.json` con metadatos de `movies.csv`, produciendo `individual_results_examples.json` con títulos legibles y resúmenes del historial del usuario.

## Modelos Disponibles

| Archivo | Tipo de modelo | Librerías | Notas relevantes |
|---------|----------------|-----------|------------------|
| `models/random_model.py` | Baseline aleatorio | NumPy | No entrena; fija semillas para reproducibilidad.
| `models/most_popular_model.py` | Popularidad global | pandas | Escala frecuencia a rango 1‑5; guarda `global_average`.
| `models/svd_model.py` | Factorización SVD (Surprise) | scikit-surprise | Usa `SVD` con `n_factors=100` y `n_epochs=20`.
| `models/item_knn_model.py` | KNN basado en ítems | scikit-surprise | Similitud coseno, `user_based=False`.
| `models/user_knn_model.py` | KNN basado en usuarios | scikit-surprise | Similitud coseno, `user_based=True`.
| `models/lightfm_model.py` | LightFM WARP | lightfm | Matriz dispersa `coo_matrix`, semillas fijadas y predicción determinista.
| `models/als_model.py` | ALS implícito | implicit | Binariza ratings ≥4, almacena mapas globales y filtra índices inválidos.
| `models/ncf_model.py` | Neural Collaborative Filtering | PyTorch | Combina GMF + MLP; DataLoader con semillas controladas.
| `models/multivae_model.py` | Variational Autoencoder | PyTorch | Binariza ratings ≥4, usa CSR y calcula novelty en log‑probabilities.
| `models/lightgcn_model.py` | LightGCN puro | PyTorch | Entrenamiento BPR con propagación multinivel; inferencia individual no soportada.

Todos los modelos exponen tres funciones: `preprocess_data`, `train_model` y `generate_predictions`, requisito para que `run_experiment_saves.py` los orqueste.

## Métricas Calculadas

- **Rendimiento**
    - `rmse`: Error cuadrático medio sobre `test.csv` vs predicciones (cuando existen ambos).
    - `precision_at_10` y `recall_at_10`: Calculadas por usuario con `k=10` y umbral de relevancia 4.0.
    - `ndcg_at_10`: Ganancia logarítmica normalizada al tope 10.
    - `hit_rate_at_10`: Fracción de usuarios con al menos un acierto (añadido por `evaluate_new_results.py`).
    - `map_at_10`, `mrr_at_10`: Media aritmética de precisión acumulada y recíproca del ranking.
    - `novelty_at_10`: Información promedio (−log₂ popularidad) de los ítems recomendados.
    - `diversity_at_10`: 1 − similitud media entre ítems recomendados (similitud coseno sobre interacciones).

- **Huella Ambiental** (CodeCarbon)
    - `co2_emissions_g`: Emisiones estimadas de CO₂ (gramos).
    - `energy_consumed_kWh`: Energía eléctrica consumida (kWh).
    - `duration_seconds`: Duración del segmento medido.

## Archivos de Salida Relevantes

- `results.json`: Diccionario `{porcentaje -> modelo -> {training_footprint, prediction_footprint, performance_metrics}}`.
- `results_updated.json`: Extiende `results.json` con métricas derivadas (hit-rate, MAP, MRR, novelty, diversity).
- `data/<porcentaje>/<modelo>_predictions.csv`: Predicciones completas sobre `antitest.csv`.
- `trained_models/<porcentaje>/`: Modelos persistidos (`.pkl`, `.pth`, `.json`) y mapas `*_user_map.json`, `*_item_map.json`.
- `emissions.csv`: Log acumulado de CodeCarbon.
- `individual_results.json`: Resultados Top‑10 por perfil, con huella de inferencia.
- `individual_results_examples.json`: Versión enriquecida con títulos y contexto.
- `graficos/`: Carpeta de PNG con métricas vs dataset, trade-offs y comparativas 3D.
- `analisis/`: CSV y TXT con porcentajes de cambio, eficiencia marginal y hallazgos narrativos.

## Flujo de Trabajo Sugerido

1. Preparar entorno y dependencias.
2. Procesar MovieLens (`preprocessdata.py`) y generar subconjuntos (`prepare_datasets.py`).
3. Ejecutar `run_all_experiments.sh` o llamadas individuales a `run_experiment_saves.py`.
4. Enriquecer métricas con `evaluate_new_results.py` y generar gráficos (`generacion_graficos.py`).
5. Auditar usuarios tipo con `test_all_individual_predictions.sh` y crear la versión legible (`create_readable.py`).
6. Revisar reportes en `analisis/` y visualizaciones en `graficos/` para documentar hallazgos.

## Aceder a dataset Books Crossing
Se debe cambiar de la rama main a la rama Books la cual realiza los mismos experimentos pero adaptados a ese dataset.

## Referencias
Como apoyo para la realizacion de este codigo se hizo uso del agente de IA GPT codex 5 integrado en VSC