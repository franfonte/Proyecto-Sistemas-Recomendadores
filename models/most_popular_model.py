import pandas as pd
import os
import sys

def preprocess_data(data_path):
    """
    Carga los datos de entrenamiento y el conjunto antitest.
    """
    print(f"1. Preprocesando datos desde: {data_path}")
    train_file = os.path.join(data_path, 'train.csv')
    antitest_file = os.path.join(data_path, 'antitest.csv')
    
    train_df = pd.read_csv(train_file)
    antitest_df = pd.read_csv(antitest_file)
    
    print("   Datos cargados en DataFrames.")
    return train_df, antitest_df

def train_model(train_df):
    """
    "Entrena" calculando la FRECUENCIA de cada película y normalizándola a una escala 1-5.
    Devuelve una serie de pandas con movieId como índice y el score de popularidad (1-5) como valor.
    """
    print("2. Calculando popularidad (Frecuencia normalizada a escala 1.0 - 5.0)...")
    
    # 1. Contar cuántas veces aparece cada película (Frecuencia real)
    # Esto define el VERDADERO ranking de popularidad.
    raw_counts = train_df.groupby('movieId')['rating'].count()
    
    # 2. Normalización Min-Max para forzar el formato de "Ratings" (1.0 a 5.0)
    # Así cumplimos con el requisito de formato sin perder el orden por popularidad.
    max_views = raw_counts.max()
    min_views = raw_counts.min()
    
    if max_views == min_views:
        # Caso borde: si todas tienen las mismas vistas o hay 1 sola película
        popularity_model = raw_counts.apply(lambda x: 5.0)
    else:
        # Fórmula: Scaled = 1 + (x - min) * (5 - 1) / (max - min)
        popularity_model = 1.0 + (raw_counts - min_views) * 4.0 / (max_views - min_views)
    
    # 3. Calcular el valor de relleno (global_average)
    # Si una película no está en train, significa que tiene 0 vistas.
    # Por lógica, es la MENOS popular posible, así que le asignamos el mínimo (1.0).
    global_average = 1.0
    
    print(f"   Modelo de popularidad creado. Item más visto escalado a 5.0, menos visto a 1.0.")
    return popularity_model, global_average

def generate_predictions(model, antitest_df):
    """
    Asigna el score de popularidad (escala 1-5) de cada película como la predicción.
    """
    print("3. Generando predicciones basadas en popularidad...")
    popularity_scores, global_average = model

    # Crea una copia para trabajar sobre ella
    predictions_df = antitest_df.copy()

    # Usa .map para asignar el score calculado. Si una película en el antitest
    # no estaba en el train set, se le asigna el valor mínimo (global_average = 1.0).
    predictions_df['prediction'] = predictions_df['movieId'].map(popularity_scores).fillna(global_average)
    
    print(f"   Se generaron {len(predictions_df)} predicciones.")
    return predictions_df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Por favor, proporciona el porcentaje del dataset (ej. 10, 25, ...).")
        sys.exit(1)
    
    dataset_percentage = sys.argv[1]
    DATA_PATH = os.path.join('data', dataset_percentage)
    
    # 1. Preprocesar datos
    train_data, antitest_data = preprocess_data(DATA_PATH)
    
    # 2. Entrenar el modelo
    trained_model = train_model(train_data)
    
    # 3. Generar predicciones
    predictions = generate_predictions(trained_model, antitest_data)
    
    # Guardar el resultado para que el script de evaluación lo pueda leer
    # (Asumiendo que quieres sobrescribir el archivo defectuoso anterior)
    output_file = os.path.join(DATA_PATH, 'most_popular_model_predictions.csv')
    predictions.to_csv(output_file, index=False)
    print(f"   Archivo guardado en: {output_file}")
    
    print("\n--- Proceso del modelo de más populares finalizado ---")
    print("Ejemplo de 5 predicciones (Score de Popularidad):")
    print(predictions.head())