# Proyecto Sistemas de Recomendación de Libros con Medición de Huella de Carbono

Este repositorio coordina experimentos de recomendación basados en el dataset Book-Crossing y mide el impacto ambiental de cada fase con CodeCarbon. Contiene pipelines para preparar datos con ratings en escala 1-10, entrenar modelos heterogéneos, registrar huellas energéticas, auditar recomendaciones individuales y generar análisis numéricos y gráficos.

## Características Clave

- Huella de CO₂, energía y duración tanto en entrenamiento como en inferencia, registrada mediante CodeCarbon.
- Catálogo de 10 modelos (filtrado colaborativo clásico, factorización, enfoques neuronales, baselines y sistemas implícitos) ubicados en `models/`.
- Automatización para lanzar todos los modelos sobre subconjuntos de 10 %, 25 %, 50 %, 75 % y 100 % con `run_all_experiments.sh`.
- Evaluaciones que calculan métricas de ranking (Precision@10, Recall@10, nDCG@10, MAP@10, Hit-Rate, MRR) y métricas de diversidad/novelty adaptadas a la escala 1-10.
- Scripts para auditar perfiles tipo, filtrar libros ya leídos, crear salidas legibles con títulos y medir el costo energético de cada listado Top-10.

## Panorama del Repositorio

```
.
├── README.md
├── requirements.txt                # Dependencias principales
├── prepare_datasets_bx.py          # Procesa Book-Crossing y genera subconjuntos estratificados 1-10
├── run_experiment_saves.py         # Orquesta entrenamiento + predicción + métricas + huella
├── run_all_experiments.sh          # Lanza run_experiment_saves.py para todos los modelos y datasets
├── test_all_individual_predictions.sh # Ejecuta get_top10_recommendations.py para perfiles tipo
├── evaluate_results.py             # RMSE y métricas@10 con umbral de relevancia 8.0
├── evaluate_new_results.py         # Amplía results.json con hit-rate, MAP, MRR, novelty, diversity
├── get_top10_recommendations.py    # Predicciones Top-10 para un usuario, con huella energética
├── predict_single.py               # Verifica que una predicción individual coincida con el batch
├── create_readable.py              # Convierte IDs a títulos (Books.csv), genera ejemplos legibles
├── generacion_graficos.py          # Visualizaciones a partir de results_updated.json
├── analisis_numericos_resultados*.py # Reportes y tablas en analisis/
├── data/
│   ├── bx/                         # Ratings.csv, Books.csv, Users.csv (fuente Book-Crossing)
│   └── {10,25,50,75,100}/          # Subconjuntos con ratings 1-10, splits y antitest
├── models/                         # Implementaciones de algoritmos de recomendación
├── trained_models/                 # Modelos entrenados y mapas usuario/ítem por porcentaje
├── results*.json                   # Métricas y huellas agregadas por modelo y dataset
├── individual_results*.json        # Auditorías Top-10 por usuario tipo (IDs y versión legible)
├── graficos/                       # Gráficos agregados (general, familia, individuales)
└── analisis/                       # Tablas y hallazgos exportados
```

## Requisitos Previos

- Python 3.10 o superior.
- Dataset Book-Crossing descargado desde [Kaggle](https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset). Se esperan los archivos `Books.csv`, `Ratings.csv` y `Users.csv` renombrados y ubicados en `data/bx/`, separados por `;` y codificados en latin-1.
- Dependencias listadas en `requirements.txt`.

## Configuración del Entorno

```bash
python3 -m venv venv
source venv/bin/activate              # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Preparación de Datos

1. Crea la carpeta `data/bx/` y coloca los archivos originales del dataset Book-Crossing renombrados como `Books.csv`, `Ratings.csv` y `Users.csv`. Los scripts esperan que estén separados por `;` y codificados en latin-1.
2. Genera subconjuntos estratificados (10 %, 25 %, 50 %, 75 %, 100 %) junto con los splits `train/test` y un antitest muestreado:
   ```bash
   python prepare_datasets_bx.py
   ```
   El script conserva la escala 1-10, elimina ratings implícitos (valor 0), filtra usuarios con menos de 2 interacciones y crea archivos `ratings.csv`, `train.csv`, `test.csv` y `antitest.csv` para cada porcentaje.

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
- Calcula RMSE, Precision@10, Recall@10 y nDCG@10 usando `evaluate_results.py` con umbral de relevancia 8.0 sobre la escala 1-10.
- Actualiza `results.json` con métricas y huellas, serializando a JSON seguro para tipos NumPy.

### Ejecución para Todos los Modelos y Subconjuntos

```bash
bash run_all_experiments.sh
```

El guion recorre las listas `MODELS` y `PERCENTAGES`. Modifícalas para incluir/excluir modelos o porcentajes. Las fallas de un modelo no detienen el resto del experimento.

## Evaluación y Visualizaciones

- `evaluate_results.py` se ejecuta dentro de `run_experiment_saves.py` para obtener RMSE, Precision@10, Recall@10 y nDCG@10 con `relevance_threshold=8.0`.
- `evaluate_new_results.py` añade Hit-Rate@10, MAP@10, MRR@10, Novelty@10 (bits) y Diversity@10 (1 - similitud coseno) empleando popularidad, matrices usuario-libro y similitud coseno. Ejemplo:
  ```bash
  python evaluate_new_results.py \
      --results results.json \
      --data-dir data \
      --output results_updated.json
  ```
- `generacion_graficos.py` transforma `results_updated.json` en gráficos (líneas, trade-offs, 3D) dentro de `graficos/general/` y promedios por familia en `graficos/familia/`.
- `analisis_numericos_resultados.py` y `analisis_numericos_resultados_individuales.py` exportan CSV y reportes `.txt` con porcentajes de cambio, eficiencia energética y análisis por perfil en `analisis/`.

## Predicciones Individuales y Auditoría

- `get_top10_recommendations.py` genera un Top-10 por usuario:
  ```bash
  python get_top10_recommendations.py \
      --model_name ncf_model \
      --dataset_percentage 100 \
      --user_id 4169 \
      --user_category "Power User"
  ```
  Acciones clave:
  - Carga el modelo entrenado y sus mapas (`trained_models/<porcentaje>/`).
  - Predice puntuaciones para todos los ISBN, filtra los libros ya leídos (según `train.csv`) y mide la huella de esa inferencia con CodeCarbon.
  - Agrega resultados a `individual_results.json` (organizado por porcentaje, modelo y perfil) con emisiones, energía y duración.

- `test_all_individual_predictions.sh` automatiza la ejecución anterior para una lista de perfiles (`USER_DATA`) y porcentajes (`DATASET_PERCS`). Ejecuta todos los modelos compatibles y omite `lightgcn_model`, que no soporta inferencia individual en este flujo.

- `predict_single.py` contrasta la predicción puntual (`user_id`, `isbn`) con el batch almacenado en `data/<porcentaje>/<modelo>_predictions.csv`, útil para depuración.

- `create_readable.py` enriquece `individual_results.json` con metadatos de `Books.csv` y `Ratings.csv`, produciendo `individual_results_examples.json` con títulos legibles y resúmenes del historial del usuario.

## Modelos Disponibles

| Archivo | Tipo de modelo | Librerías | Notas relevantes (escala 1-10) |
|---------|----------------|-----------|--------------------------------|
| `models/random_model.py` | Baseline aleatorio | NumPy | Predicciones uniformes en [1, 10], semillas fijadas. |
| `models/most_popular_model.py` | Promedio global | pandas | Devuelve media de rating por ISBN y promedio global. |
| `models/svd_model.py` | Factorización SVD (Surprise) | scikit-surprise | `Reader(rating_scale=(1, 10))`, `n_factors=100`, `n_epochs=20`. |
| `models/item_knn_model.py` | KNN basado en ítems | scikit-surprise | Similitud coseno, escala 1-10 preservada. |
| `models/user_knn_model.py` | KNN basado en usuarios | scikit-surprise | Similitud coseno, escala 1-10 preservada. |
| `models/lightfm_model.py` | LightFM WARP | lightfm | Matriz COO con ratings 1-10, semillas fijadas para determinismo. |
| `models/als_model.py` | ALS implícito | implicit | Binariza ratings ≥ 8 como positivos, usa mapas globales y CSR. |
| `models/ncf_model.py` | Neural Collaborative Filtering | PyTorch | Entrena con ratings 1-10, combina GMF + MLP, DataLoader reproducible. |
| `models/multivae_model.py` | Variational Autoencoder | PyTorch | Usa CSR 1-10, convierte ratings ≥ 8 a feedback positivo. |
| `models/lightgcn_model.py` | LightGCN puro | PyTorch | Grafo bipartito con interacciones ≥ 8, entrenamiento BPR; inferencia individual pendiente. |

Todos los modelos exponen tres funciones: `preprocess_data`, `train_model` y `generate_predictions`, requisito para que `run_experiment_saves.py` los orqueste.

## Métricas Calculadas

- **Rendimiento**
  - `rmse`: error cuadrático medio entre `test.csv` y las predicciones (cuando existen ambos conjuntos).
  - `precision_at_10`, `recall_at_10` y `ndcg_at_10`: métricas por usuario con `k=10` y relevancia definida como rating ≥ 8 (escala 1-10).
  - `hit_rate_at_10`: fracción de usuarios con al menos un acierto (añadido por `evaluate_new_results.py`).
  - `map_at_10`, `mrr_at_10`: precisión media acumulada y recíproca.
  - `novelty_at_10`: bits de información promedio (`-log₂` popularidad) de los ISBN recomendados.
  - `diversity_at_10`: 1 menos la similitud coseno promedio entre recomendaciones.

- **Huella Ambiental** (CodeCarbon)
  - `co2_emissions_g`: emisiones estimadas de CO₂ (gramos).
  - `energy_consumed_kWh`: energía eléctrica consumida (kWh).
  - `duration_seconds`: duración del segmento medido.

## Archivos de Salida Relevantes

- `results.json`: diccionario `{porcentaje -> modelo -> {training_footprint, prediction_footprint, performance_metrics}}`.
- `results_updated.json`: extiende `results.json` con métricas derivadas (hit-rate, MAP, MRR, novelty, diversity).
- `data/<porcentaje>/<modelo>_predictions.csv`: predicciones completas sobre `antitest.csv` (ISBN, userId, score).
- `trained_models/<porcentaje>/`: modelos persistidos (`.pkl`, `.pth`, `.json`) y mapas `*_user_map.json`, `*_item_map.json` (ID crudo → índice interno).
- `emissions.csv`: log acumulado de CodeCarbon.
- `individual_results.json`: resultados Top-10 por perfil con huella de inferencia.
- `individual_results_examples.json`: versión enriquecida con títulos de libros y contexto.
- `graficos/`: PNG con métricas vs dataset, trade-offs y comparativas 3D.
- `analisis/`: CSV y TXT con porcentajes de cambio, eficiencia marginal y hallazgos narrativos.

## Flujo de Trabajo Sugerido

1. Crear el entorno virtual e instalar dependencias.
2. Descargar Book-Crossing, renombrar archivos y colocarlos en `data/bx/`.
3. Ejecutar `python prepare_datasets_bx.py` para generar los subconjuntos estratificados y el antitest.
4. Entrenar modelos con `run_experiment_saves.py` o lanzar todos mediante `run_all_experiments.sh`.
5. Enriquecer métricas con `evaluate_new_results.py` y generar visualizaciones con `generacion_graficos.py`.
6. Auditar perfiles con `test_all_individual_predictions.sh`, crear la versión legible con `create_readable.py` y revisar reportes y gráficos en `analisis/` y `graficos/`.