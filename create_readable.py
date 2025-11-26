import json
import pandas as pd
import os
import re

# --- CONFIGURACIÓN ---
# Asegúrate de que el nombre del archivo de entrada sea el correcto (el que tiene los IDs numéricos)
# Si tu archivo original se llama diferente, cambia 'audit_results.json' por el nombre real.
INPUT_JSON = 'individual_results.json' 

# Nombre del archivo de salida solicitado
OUTPUT_JSON = 'individual_results_examples.json'

# Rutas a los datos de MovieLens (Ajusta si tus carpetas son diferentes)
MOVIES_PATH = 'data/bx/Books.csv'
RATINGS_PATH = 'data/bx/Ratings.csv'

def load_metadata():
    """Carga el mapa de ID a Título de la película."""
    print(f"Cargando metadatos desde {MOVIES_PATH}...")
    try:
        # Intentamos leer con encoding común para MovieLens
        # Usamos 'latin-1' porque a veces hay caracteres especiales en los títulos
        df_movies = pd.read_csv(
            MOVIES_PATH,
            sep=';',
            encoding='utf-8'
        )
        
        # Aseguramos que los IDs sean strings para coincidir con las llaves del JSON
        df_movies['ISBN'] = df_movies['ISBN'].astype(str)
        
        # Creamos diccionario: {'1': 'Toy Story (1995)', '2': 'Jumanji (1995)', ...}
        id_to_title = pd.Series(df_movies.Title.values, index=df_movies.ISBN).to_dict()
        return id_to_title
    except Exception as e:
        print(f"Error cargando movies.csv: {e}")
        return {}

def get_user_history_titles(user_id_str, df_ratings, id_to_title, limit=15):
    """
    Extrae las películas que el usuario YA vio, ordenadas por rating (lo que más le gustó).
    Retorna una lista de títulos para dar contexto de sus gustos.
    """
    # Filtramos el historial del usuario específico
    user_history = df_ratings[df_ratings['User-ID'] == user_id_str]
    
    if user_history.empty:
        return ["No history found in ratings.csv"]

    # Ordenamos: Primero las de 5 estrellas, luego 4, etc.
    # Si existe timestamp, se usa para desempatar (las más recientes primero)
    sort_cols = ['Rating']
    ascending_vals = [False]
    if 'timestamp' in df_ratings.columns:
        sort_cols.append('timestamp')
        ascending_vals.append(False)
        
    user_history = user_history.sort_values(by=sort_cols, ascending=ascending_vals)
    
    # Tomamos solo los IDs del top N historial
    top_history_ids = user_history['ISBN'].head(limit).values
    
    # Mapeamos IDs -> Títulos reales
    titles = [id_to_title.get(str(mid), f"Unknown Movie ID: {mid}") for mid in top_history_ids]
    return titles

def map_recommendations(rec_ids, id_to_title):
    """Convierte una lista de IDs de recomendación a sus Títulos."""
    return [id_to_title.get(str(rid), f"Unknown Movie ID: {rid}") for rid in rec_ids]

def extract_user_id_from_key(key_string):
    """
    Extrae el ID numérico de la llave compuesta del JSON.
    Ejemplo: "4169 (Power User)" -> "4169"
    """
    match = re.match(r"(\d+)", str(key_string))
    if match:
        return match.group(1)
    return None

def main():
    # 1. Verificar archivos
    if not os.path.exists(INPUT_JSON):
        print(f"❌ Error: No se encontró el archivo de entrada '{INPUT_JSON}'.")
        print("Asegúrate de haber generado primero el JSON con los resultados de auditoría.")
        return

    if not os.path.exists(MOVIES_PATH) or not os.path.exists(RATINGS_PATH):
        print(f"❌ Error: No se encuentran los archivos CSV en 'data/bx/'.")
        return

    # 2. Cargar Mapeos (ID -> Titulo)
    id_to_title = load_metadata()
    
    # 3. Cargar Ratings (Historial)
    print(f"Cargando historial completo de ratings desde {RATINGS_PATH}...")
    # Optimizamos cargando solo columnas necesarias y forzando tipos string para IDs
    df_ratings = pd.read_csv(
        RATINGS_PATH,
        sep=';',
        usecols=['User-ID', 'ISBN', 'Rating'],
        dtype={'User-ID': str, 'ISBN': str}
    )
    
    # 4. Cargar JSON original
    print(f"Leyendo {INPUT_JSON}...")
    with open(INPUT_JSON, 'r') as f:
        audit_data = json.load(f)
    
    print("Procesando y enriqueciendo datos...")
    
    # 5. Estructura del nuevo JSON
    new_audit_data = {}
    
    # Iteramos: Tamaño -> Modelo -> Usuario
    for size, models in audit_data.items():
        new_audit_data[size] = {}
        for model, users in models.items():
            new_audit_data[size][model] = {}
            for user_key, user_data in users.items():
                
                # Extraer el ID puro del usuario (ej: "4169")
                raw_user_id = extract_user_id_from_key(user_key)
                
                # A. Traducir Recomendaciones (Top 10 IDs -> Títulos)
                raw_recs = user_data.get('top_10_recommendations', [])
                # Aseguramos que sean strings para el mapeo
                readable_recs = map_recommendations([str(r) for r in raw_recs], id_to_title)
                
                # B. Obtener Historial Real (Top 15 películas que amó)
                if raw_user_id:
                    history_titles = get_user_history_titles(raw_user_id, df_ratings, id_to_title, limit=15)
                else:
                    history_titles = ["Error extracting User ID"]

                # Construir el nuevo bloque para este usuario
                new_audit_data[size][model][user_key] = {
                    "user_profile_summary": f"User {raw_user_id} - Top Rated History", # Etiqueta descriptiva
                    "user_history_top_rated": history_titles,       # NUEVO: Qué le gusta realmente
                    "top_10_recommendations": readable_recs,        # MODIFICADO: Nombres en vez de IDs
                    "inference_footprint": user_data.get('inference_footprint', {}) # Mantenemos la huella
                }

    # 6. Guardar el resultado final
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(new_audit_data, f, indent=4, ensure_ascii=False)
    
    print(f"✅ ¡Éxito! Archivo generado: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()