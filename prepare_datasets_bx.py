import pandas as pd
import os
import sys
import random
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split as surprise_train_test_split

# --- Configuración Global (MODIFICADA) ---
INPUT_DATA_PATH = os.path.join('data', 'bx') # Ruta para LEER el original
OUTPUT_DATA_PATH = 'data' # Ruta para GUARDAR las carpetas 10, 25, etc.
ORIGINAL_RATINGS_FILE = os.path.join(INPUT_DATA_PATH, 'Ratings.csv') 
RANDOM_SEED = 42
MAX_ANTITEST_SIZE = 20_000_000

def create_dataset_subsets():
    """
    (MODIFICADA)
    Lee el Ratings.csv de 'data/bx', lo procesa, y guarda los subconjuntos
    directamente en 'data/10', 'data/25', etc.
    """
    if not os.path.exists(ORIGINAL_RATINGS_FILE):
        print(f"Error: Archivo de ratings original no encontrado en '{ORIGINAL_RATINGS_FILE}'")
        sys.exit(1)

    print("--- Iniciando pre-procesamiento de Book-Crossings ---")
    
    try:
        df = pd.read_csv(
            ORIGINAL_RATINGS_FILE,
            sep=';',
            encoding='latin-1',
            on_bad_lines='skip'
        )
    except Exception as e:
        print(f"Error al leer el CSV de Book-Crossings: {e}")
        try:
            print("Intentando con encoding 'utf-8'...")
            df = pd.read_csv(
                ORIGINAL_RATINGS_FILE,
                sep=';',
                encoding='utf-8',
                on_bad_lines='skip'
            )
        except Exception as e_utf8:
            print(f"Error también con utf-8: {e_utf8}")
            sys.exit(1)

    expected_cols = ['User-ID', 'ISBN', 'Rating'] 
    if not all(col in df.columns for col in expected_cols):
        print(f"Error: Faltan columnas. Se esperaban {expected_cols}, pero se encontraron {df.columns}")
        sys.exit(1)

    df_explicit = df[df['Rating'] != 0].copy()
    df_explicit['Rating'] = df_explicit['Rating'].astype(int)
    df_processed = df_explicit[['User-ID', 'ISBN', 'Rating']]
    df_processed.columns = ['userId', 'movieId', 'rating']
    
    print(f"Dataset procesado: {len(df_processed)} ratings explícitos (escala 1-10)")

    # Filtrar usuarios con < 2 ratings para permitir la estratificación
    print("Filtrando usuarios con < 2 ratings para permitir la estratificación...")
    user_counts = df_processed['userId'].value_counts()
    users_to_keep = user_counts[user_counts >= 2].index
    df_stratify_ready = df_processed[df_processed['userId'].isin(users_to_keep)]
    
    print(f"Dataset listo para estratificar: {len(df_stratify_ready)} ratings (de {len(users_to_keep)} usuarios)")
    
    print("--- Iniciando creación de subconjuntos ---")

    percentages = [10, 25, 50, 75, 100]

    for p in percentages:
        # Los subconjuntos se guardan directamente en 'data/'
        subset_dir = os.path.join(OUTPUT_DATA_PATH, str(p)) # <-- CORREGIDO
        os.makedirs(subset_dir, exist_ok=True)
        output_path = os.path.join(subset_dir, 'ratings.csv')

        if p == 100:
            subset_df = df_stratify_ready 
        else:
            subset_df, _ = sklearn_train_test_split(
                df_stratify_ready,
                train_size=(p / 100.0),
                stratify=df_stratify_ready['userId'], 
                random_state=RANDOM_SEED
            )
        
        subset_df.to_csv(output_path, index=False)
        print(f"Successfully created '{output_path}' with {len(subset_df)} ratings ({p}%)")
    
    print("--- Todos los subconjuntos creados exitosamente ---\n")


def split_and_generate_antitest(subset_dir_path):
    """
    (Sin cambios)
    Toma un ratings.csv, lo divide en 80/20 train/test, y genera un 
    antitest MUESTREADO (max 20M de líneas) proporcionalmente por usuario.
    """
    ratings_file = os.path.join(subset_dir_path, 'ratings.csv')
    print(f"--- Procesando directorio: {subset_dir_path} ---")

    # 1. Cargar datos con Surprise
    df = pd.read_csv(ratings_file)
    reader = Reader(rating_scale=(1, 10))
    df = df.dropna(subset=['userId', 'movieId', 'rating'])
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    
    # 2. Split en train y test (80/20)
    trainset, testset = surprise_train_test_split(data, test_size=0.20, random_state=RANDOM_SEED)

    # 3. Generar anti-test set (LÓGICA DE MUESTREO)
    print(f"  -> Construyendo antitest MUESTREADO (max {MAX_ANTITEST_SIZE} líneas)...")
    
    n_users = trainset.n_users
    samples_per_user = max(1, int(MAX_ANTITEST_SIZE / n_users))
    
    print(f"  -> {n_users} usuarios en trainset. Muestreando ~{samples_per_user} negativos por usuario.")

    anti_testset = []
    all_item_inner_ids = set(trainset.all_items()) 
    random.seed(RANDOM_SEED) 

    for user_inner_id in trainset.all_users():
        items_rated_by_user = set([item_inner_id for (item_inner_id, rating) in trainset.ur[user_inner_id]])
        items_not_rated = list(all_item_inner_ids - items_rated_by_user)
        
        if not items_not_rated:
            continue 
            
        num_to_sample = min(samples_per_user, len(items_not_rated))
        sampled_item_inner_ids = random.sample(items_not_rated, num_to_sample)
        
        raw_uid = trainset.to_raw_uid(user_inner_id)
        for item_inner_id in sampled_item_inner_ids:
            raw_iid = trainset.to_raw_iid(item_inner_id)
            anti_testset.append((raw_uid, raw_iid, 0.0)) 

    print(f"  -> Antitest muestreado construido con {len(anti_testset)} pares.")

    # 4. Convertir de nuevo a DataFrames y guardar
    # Train set
    train_df = pd.DataFrame(trainset.all_ratings(), columns=['userId_inner', 'movieId_inner', 'rating'])
    train_df['userId'] = train_df['userId_inner'].apply(trainset.to_raw_uid)
    train_df['movieId'] = train_df['movieId_inner'].apply(trainset.to_raw_iid)
    train_df[['userId', 'movieId', 'rating']].to_csv(os.path.join(subset_dir_path, 'train.csv'), index=False)
    
    # Test set
    test_df = pd.DataFrame(testset, columns=['userId', 'movieId', 'rating'])
    test_df.to_csv(os.path.join(subset_dir_path, 'test.csv'), index=False)

    # Anti-test set (muestreado)
    anti_test_df = pd.DataFrame(anti_testset, columns=['userId', 'movieId', 'rating_placeholder'])
    anti_test_df['rating'] = '' 
    anti_test_df = anti_test_df[['userId', 'movieId', 'rating']] 
    
    print(f"  -> Guardando antitest.csv...")
    anti_test_df.to_csv(os.path.join(subset_dir_path, 'antitest.csv'), index=False)

    print(f"  -> Creados train.csv, test.csv, y antitest.csv")

if __name__ == "__main__":
    # Step 1: Crear los 5 directorios
    create_dataset_subsets()

    # Step 2: Para cada subconjunto, dividir y crear antitest MUESTREADO
    percentages = [10, 25, 50, 75, 100]
    for p in percentages:
        # Rutas actualizadas para apuntar a 'data/10', 'data/25', etc.
        subset_dir = os.path.join(OUTPUT_DATA_PATH, str(p)) # <-- CORREGIDO
        split_and_generate_antitest(subset_dir)
        
    print("\n--- Todos los datasets de Book-Crossings han sido preparados (con antitest muestreado)! ---")