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
    "Entrena" calculando la calificación promedio de cada película.
    Devuelve una serie de pandas con movieId como índice y la calificación promedio como valor.
    """
    print("2. Calculando la popularidad de las películas (calificación promedio)...")
    
    # Calcular la calificación promedio para cada movieId
    popularity_model = train_df.groupby('movieId')['rating'].mean()
    
    # Calcular el promedio global para rellenar en caso de películas no vistas en train
    global_average = train_df['rating'].mean()
    
    print(f"   Modelo de popularidad creado. Promedio global: {global_average:.2f}")
    return popularity_model, global_average

def generate_predictions(model, antitest_df):
    """
    Asigna la calificación promedio de cada película como la predicción.
    """
    print("3. Generando predicciones basadas en popularidad...")
    popularity_scores, global_average = model

    # Crea una copia para trabajar sobre ella
    predictions_df = antitest_df.copy()

    # Usa .map para asignar la calificación promedio. Si una película en el antitest
    # no estaba en el train set, se le asigna el promedio global.
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
    
    print("\n--- Proceso del modelo de más populares finalizado ---")
    print("Ejemplo de 5 predicciones:")
    print(predictions.head())
