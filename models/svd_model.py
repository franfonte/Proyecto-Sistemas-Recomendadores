import pandas as pd
import os
import sys
from surprise import SVD, Dataset, Reader

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
    
    # Cargar el conjunto de entrenamiento
    train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    trainset = train_data.build_full_trainset()
    
    # El antitest set se convierte a una lista de tuplas para la predicción
    antitest_tuples = [tuple(x) for x in antitest_df[['userId', 'movieId', 'rating']].to_numpy()]

    print("   Datos cargados y listos para Surprise.")
    return trainset, antitest_tuples

def train_model(trainset):
    """
    Entrena un modelo SVD con el conjunto de datos de entrenamiento.
    """
    print("2. Entrenando el modelo SVD...")
    algo = SVD(n_factors=100, n_epochs=20, random_state=42)
    algo.fit(trainset)
    print("   Entrenamiento completado.")
    return algo

def generate_predictions(model, antitest_set):
    """
    Genera predicciones para el conjunto antitest.
    """
    print("3. Generando predicciones con SVD...")
    predictions = model.test(antitest_set)
    
    # Convertir las predicciones a un DataFrame de Pandas
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
    
    print("\n--- Proceso del modelo SVD finalizado ---")
    print("Ejemplo de 5 predicciones:")
    print(predictions.head())
