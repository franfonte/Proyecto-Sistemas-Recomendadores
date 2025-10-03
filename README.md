# Proyecto Sistemas de Recomendación con Medición de Huella de Carbono

Este proyecto mide la huella de carbono de Sistemas de Recomendación utilizando la biblioteca CodeCarbon para analizar el impacto ambiental de diferentes algoritmos de recomendación.

## Estructura del Proyecto

```
.
├── README.md
├── requirements.txt         # Dependencias de Python
├── analisis_datos.ipynb     # Jupyter notebook para analizar los datos
├── run_experiment.py        # Script principal para ejecutar experimentos con seguimiento de carbono
├── preprocessdata.py        # Script para convertir archivos .dat a CSV
├── prepare_datasets.py      # Script para preparar conjuntos de datos con diferentes tamaños
├── data/                    # Directorio para conjuntos de datos
│   ├── ml-1m/              # Dataset MovieLens 1M original
│   │   ├── ratings.csv     # Calificaciones en formato CSV
│   │   ├── ratings.dat     # Calificaciones en formato original
│   │   ├── movies.dat      # Información de películas
│   │   └── users.dat       # Información de usuarios
│   ├── ml-latest-small/    # Dataset MovieLens pequeño de muestra
│   ├── 10/                 # Subconjunto 10% del dataset
│   │   ├── ratings.csv     # Datos completos (10%)
│   │   ├── train.csv       # Conjunto de entrenamiento
│   │   ├── test.csv        # Conjunto de prueba
│   │   └── antitest.csv    # Conjunto anti-test
│   ├── 25/                 # Subconjunto 25% del dataset
│   ├── 50/                 # Subconjunto 50% del dataset
│   ├── 75/                 # Subconjunto 75% del dataset
│   └── 100/                # Dataset completo (100%)
└── models/                  # Directorio para scripts de modelos
    └── svd_model.py        # Modelo SVD de ejemplo
```

## Instalación

### Requisitos Previos
- Python 3.10 o superior
- Dataset MovieLens 1M (descargar desde [GroupLens](https://grouplens.org/datasets/movielens/1m/))

### Configuración del Entorno

1. **Crear y activar un entorno virtual (recomendado):**
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

2. **Instalar las dependencias:**
```bash
pip install -r requirements.txt
```

3. **Preparar los datos:**
```bash
# Convertir archivos .dat a CSV (si es necesario)
python preprocessdata.py

# Crear subconjuntos de datos con diferentes tamaños
python prepare_datasets.py
```

## Uso

### Listar modelos disponibles

```bash
python run_experiment.py --list
```

### Ejecutar un modelo con seguimiento de carbono

```bash
python run_experiment.py --model svd_model --dataset_percentage 10
```

### Ejemplos de Uso

```bash
# Activar el entorno virtual
source venv/bin/activate

# Listar todos los modelos disponibles
python run_experiment.py --list

# Ejecutar experimento con modelo SVD
python run_experiment.py --model svd_model --dataset_percentage 10

# Procesar datos desde cero
python preprocessdata.py
python prepare_datasets.py
```

## Características

- **Seguimiento de Huella de Carbono**: Utiliza CodeCarbon para medir las emisiones de CO2 durante la ejecución de modelos
- **Sistema de Modelos Flexible**: Fácil agregar nuevos modelos colocando scripts de Python en el directorio `models/`
- **Informes Detallados**: Imprime mediciones exhaustivas de emisiones después de cada experimento
- **Ejecución en Subprocesos**: Los modelos se ejecutan en subprocesos aislados para mediciones precisas
- **Múltiples Tamaños de Dataset**: Experimentos con conjuntos de datos de diferentes tamaños (10%, 25%, 50%, 75%, 100%)
- **Preparación Automática de Datos**: Scripts para convertir y preparar automáticamente los conjuntos de datos
- **Conjuntos Anti-test**: Generación automática de conjuntos anti-test para evaluación completa
- **Compatibilidad con MovieLens**: Soporte completo para datasets MovieLens 1M y small


## Dependencias

- **pandas**: Manipulación y análisis de datos
- **scikit-surprise**: Biblioteca para sistemas de recomendación
- **codecarbon**: Seguimiento de emisiones de carbono
- **psutil**: Utilidades de sistema y procesos
- **scikit-learn**: Herramientas de aprendizaje automático
- **numpy<2**: Computación numérica (versión compatible con Surprise)

## Conjuntos de Datos

El proyecto utiliza el dataset **MovieLens 1M** que contiene:
- **1,000,209 calificaciones** de 6,040 usuarios en 3,952 películas
- **Calificaciones**: Escala de 1-5 estrellas
- **Período**: Datos recolectados entre 1996-2018

### Subconjuntos Disponibles
- **10%**: 100,020 calificaciones
- **25%**: 250,052 calificaciones  
- **50%**: 500,104 calificaciones
- **75%**: 750,156 calificaciones
- **100%**: 1,000,209 calificaciones

Cada subconjunto incluye división automática en conjuntos de entrenamiento (80%), prueba (20%) y anti-test.

## Scripts Disponibles

- **`preprocessdata.py`**: Convierte archivos .dat de MovieLens a formato CSV
- **`prepare_datasets.py`**: Crea subconjuntos de datos con diferentes tamaños y divisiones
- **`run_experiment.py`**: Ejecuta experimentos con seguimiento de huella de carbono
- **`models/svd_model.py`**: Implementación de ejemplo con algoritmo SVD

## Resultados

Los experimentos generan:
- **Métricas de rendimiento**: RMSE, MAE, precisión, recall
- **Mediciones de carbono**: Emisiones de CO2, consumo energético
- **Informes detallados**: Tiempo de ejecución, uso de recursos
- **Archivos de resultados**: Logs automáticos con CodeCarbon

## Contribuciones

Para contribuir al proyecto:
1. Fork el repositorio
2. Crea una rama para tu característica
3. Implementa tus cambios
4. Agrega pruebas si es necesario
5. Envía un pull request

## Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo LICENSE para más detalles.