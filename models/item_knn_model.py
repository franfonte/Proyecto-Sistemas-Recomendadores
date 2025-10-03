import pandas as pd
import os
import sys
from surprise import KNNBasic, Dataset, Reader

def preprocess_data(data_path):
    """
    Carga los datos de train y antitest en el formato de la librería Surprise.
    """
    print(f"1. Preprocesando datos para Surprise desde: {data_path}")
    train_file = os.path.join(data_path, 'train.csv')
    antitest_file = os.path.join(data_path, 'antitest.csv')

    train_df = pd.read_csv(train_file)
    antitest_df = pd.read_csv(antitest_file)

    reader = Reader(rating_scale=(1, 5))
    
    train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    trainset = train_data.build_full_trainset()
    
    antitest_tuples = [tuple(x) for x in antitest_df[['userId', 'movieId', 'rating']].to_numpy()]

    print("   Datos cargados y listos para Surprise.")
    return trainset, antitest_tuples

def train_model(trainset):
    """
    Entrena un modelo Item-Based KNN.
    """
    print("2. Entrenando el modelo Item-Based KNN...")
    # La diferencia clave es 'user_based': False
    sim_options = {'name': 'cosine', 'user_based': False}
    algo = KNNBasic(sim_options=sim_options)
    algo.fit(trainset)
    print("   Entrenamiento completado.")
    return algo

def generate_predictions(model, antitest_set):
    """
    Genera predicciones para el conjunto antitest.
    """
    print("3. Generando predicciones con Item-Based KNN...")
    predictions = model.test(antitest_set)
    
    predictions_df = pd.DataFrame(predictions, columns=['userId', 'movieId', 'actual_rating', 'prediction', 'details'])
    predictions_df = predictions_df[['userId', 'movieId', 'prediction']]
    
    print(f"   Se generaron {len(predictions_df)} predicciones.")
    return predictions_df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Por favor, proporciona el porcentaje del dataset (ej. 10, 25, ...).")
        sys.exit(1)
    
    dataset_percentage = sys.argv[1]
    DATA_PATH = os.path.join('data', dataset_percentage)

    # 1. Preprocesar datos
    train_set, antitest_data = preprocess_data(DATA_PATH)
    
    # 2. Entrenar el modelo
    trained_model = train_model(train_set)
    
    # 3. Generar predicciones
    predictions = generate_predictions(trained_model, antitest_data)
    
    print("\n--- Proceso del modelo Item-Based KNN finalizado ---")
    print("Ejemplo de 5 predicciones:")
    print(predictions.head())
