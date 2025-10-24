import pandas as pd
import numpy as np
import os
import sys
from scipy.sparse import csr_matrix
# import implicit # No necesario en este archivo
import torch ## Agregado para chequear tipo
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#from torch_geometric.utils import to_scipy_sparse_matrix # Ya no se usa torch_geometric
import scipy.sparse as sp # Ya estaba importado
import random # Ya estaba importado
import time # Ya estaba importado
import torch.nn.functional as F # Importar F

# --- Constante para Replicabilidad ---
RANDOM_SEED = 42

# --- Función para Fijar Semillas ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"   Semillas fijadas en: {seed}")

# --- Worker Init Fn para DataLoader ---
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --- Hiperparámetros (Basados en el notebook y nuestro proyecto) ---
EMBEDDING_DIM = 64
NUM_LAYERS = 3 
LEARNING_RATE = 0.001 # LR estándar del paper
BATCH_SIZE = 1024 
EPOCHS = 30 
WEIGHT_DECAY = 1e-4 # Regularización L2 manual

# --- 1. Arquitectura del Modelo LightGCN (Implementación Pura de PyTorch) ---
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, num_layers, embedding_dim, norm_adj_matrix):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        # Embeddings iniciales (serán los únicos parámetros entrenables)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Inicialización normal (como en el notebook)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        print('   Modelo LightGCN (PyTorch Puro) inicializado con embeddings Normal(0, 0.1)')

        # Convertir la matriz SciPy a un Tensor Disperso de PyTorch
        self.norm_adj_matrix = self.convert_sp_mat_to_sp_tensor(norm_adj_matrix).coalesce()

    def convert_sp_mat_to_sp_tensor(self, X):
        """Convierte una matriz dispersa de SciPy a un Tensor disperso de PyTorch."""
        coo = X.tocoo().astype(np.float32)
        row = torch.from_numpy(coo.row).long()
        col = torch.from_numpy(coo.col).long()
        index = torch.stack([row, col])
        data = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

    def forward(self):
        """Calcula los embeddings finales después de la propagación."""
        initial_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [initial_embeddings] # Lista para acumular embeddings de capa (incluida la 0)
        current_embeddings = initial_embeddings
        device = initial_embeddings.device # Dispositivo (ej. MPS)

        # Mover la matriz al dispositivo correcto (CPU para sparse.mm en MPS)
        norm_adj_matrix_dev = self.norm_adj_matrix.to('cpu') 

        for _ in range(self.num_layers):
            # Mover embeddings a CPU para la multiplicación
            cpu_embeddings = current_embeddings.to('cpu')
            # Propagar
            propagated_embeddings = torch.sparse.mm(norm_adj_matrix_dev, cpu_embeddings)
            # Mover de vuelta al dispositivo principal
            current_embeddings = propagated_embeddings.to(device)
            all_embeddings.append(current_embeddings)

        # Combinar embeddings (media de todas las capas, como en el paper)
        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)
        
        final_user_emb, final_item_emb = torch.split(final_embeddings, [self.num_users, self.num_items])
        return final_user_emb, final_item_emb

    def bpr_loss(self, users_emb_initial, items_emb_initial, users_emb_final, items_emb_final, users, pos_items, neg_items):
        """
        Calcula la pérdida BPR + Regularización L2 manual (como en el notebook).
        """
        # 1. Pérdida de Ranking (usa embeddings FINALES)
        user_emb = users_emb_final[users]
        pos_item_emb = items_emb_final[pos_items]
        neg_item_emb = items_emb_final[neg_items]
        
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
        
        ranking_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        # 2. Pérdida de Regularización L2 (usa embeddings INICIALES)
        user_emb_0 = users_emb_initial[users]
        pos_item_emb_0 = items_emb_initial[pos_items]
        neg_item_emb_0 = items_emb_initial[neg_items]
        
        reg_loss = (user_emb_0.norm(2).pow(2) +
                    pos_item_emb_0.norm(2).pow(2) +
                    neg_item_emb_0.norm(2).pow(2)) / float(len(users))
        
        return ranking_loss, reg_loss


# --- 2. Dataset y Funciones Auxiliares ---
class BPRDataset(Dataset):
    def __init__(self, train_df, num_items, user_interactions):
        self.users = torch.from_numpy(train_df['user_idx'].values).long()
        self.pos_items = torch.from_numpy(train_df['item_idx'].values).long()
        self.num_items = num_items
        self.user_interactions = user_interactions

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx].item()
        pos_item = self.pos_items[idx].item()
        user_seen_items = self.user_interactions.get(user, set())
        neg_item = random.randint(0, self.num_items - 1)
        while neg_item in user_seen_items:
             neg_item = random.randint(0, self.num_items - 1)
        return user, pos_item, neg_item

def create_norm_adj_matrix(edge_index_np, num_users, num_items):
    """
    Crea la matriz de adyacencia normalizada simétricamente (D^-1/2 * A * D^-1/2).
    Devuelve una matriz dispersa de SciPy.
    """
    user_nodes = edge_index_np[0]
    item_nodes = edge_index_np[1]
    
    # Crear grafo bipartito
    rows = np.concatenate([user_nodes, item_nodes + num_users])
    cols = np.concatenate([item_nodes + num_users, user_nodes])
    data = np.ones(len(rows))
    
    adj_matrix = sp.coo_matrix((data, (rows, cols)), shape=(num_users + num_items, num_users + num_items), dtype=np.float32)
    
    # Normalización
    row_sum = np.array(adj_matrix.sum(axis=1)).flatten()
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    norm_adj_matrix = d_mat_inv_sqrt.dot(adj_matrix).dot(d_mat_inv_sqrt).tocoo()
    
    print("   Matriz de adyacencia normalizada creada (formato SciPy COO).")
    return norm_adj_matrix

# --- 3. Funciones para la Integración con run_experiment.py ---
def preprocess_data(data_path):
    """
    Preprocesa datos para LightGCN.
    ¡CORREGIDO! Binariza los datos (>= 4.0) y crea la matriz dispersa.
    """
    print(f"1. Preprocesando datos para LightGCN (Implementación Corregida) desde: {data_path}")
    train_file = os.path.join(data_path, 'train.csv')
    antitest_file = os.path.join(data_path, 'antitest.csv')
    train_df = pd.read_csv(train_file)
    antitest_df = pd.read_csv(antitest_file)

    # Mapas globales
    all_users = pd.concat([train_df['userId'], antitest_df['userId']]).unique()
    all_items = pd.concat([train_df['movieId'], antitest_df['movieId']]).unique()
    user_map = {uid: i for i, uid in enumerate(all_users)}
    item_map = {iid: i for i, iid in enumerate(all_items)}
    num_users, num_items = len(user_map), len(item_map)
    print(f"   Usuarios únicos totales (train+antitest): {num_users}")
    print(f"   Items únicos totales (train+antitest): {num_items}")

    # <<<<<<< CORRECCIÓN CRÍTICA: Binarizar/Filtrar train_df >>>>>>>
    print("   Aplicando regla de binarización: rating >= 4.0 --> 1 (Feedback Positivo)")
    train_positive = train_df[train_df['rating'] >= 4.0].copy()
    
    # Mapear train_positive
    train_positive['user_idx'] = train_positive['userId'].map(user_map)
    train_positive['item_idx'] = train_positive['movieId'].map(item_map)
    train_positive = train_positive.dropna(subset=['user_idx', 'item_idx'])
    train_positive['user_idx'] = train_positive['user_idx'].astype(int)
    train_positive['item_idx'] = train_positive['item_idx'].astype(int)
    print(f"   Interacciones positivas (>=4) a usar para grafo: {len(train_positive)}")

    # Crear interacciones y edge_index (basado en train_positive)
    user_interactions = train_positive.groupby('user_idx')['item_idx'].apply(set).to_dict()
    edge_index_np = np.vstack([train_positive['user_idx'].values, train_positive['item_idx'].values])
    
    # Crear la matriz de adyacencia normalizada (en formato SciPy)
    norm_adj_matrix_scipy = create_norm_adj_matrix(edge_index_np, num_users, num_items)

    # Crear DataLoader (usa train_positive)
    train_dataset = BPRDataset(train_positive, num_items, user_interactions)
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, worker_init_fn=seed_worker, generator=g
    )

    # Agrupar componentes
    training_components = {
        'train_loader': train_loader,
        'num_users': num_users,
        'num_items': num_items,
        'norm_adj_matrix': norm_adj_matrix_scipy # Pasar la matriz SciPy
    }
    prediction_components = { 
        'antitest_df': antitest_df,
        'norm_adj_matrix': norm_adj_matrix_scipy # Pasar también para la predicción
    }
    return training_components, prediction_components, user_map, item_map

def train_model(training_components):
    """
    Entrena el modelo LightGCN (Puro PyTorch) con BPR + L2 Manual.
    """
    set_seed(RANDOM_SEED)
    print("2. Entrenando el modelo LightGCN (Puro PyTorch Corregido)...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Usando dispositivo: {device}")

    train_loader = training_components['train_loader']
    num_users = training_components['num_users']
    num_items = training_components['num_items']
    norm_adj_matrix = training_components['norm_adj_matrix'] # Viene como SciPy

    model = LightGCN(num_users, num_items, NUM_LAYERS, EMBEDDING_DIM, norm_adj_matrix).to(device)
    
    # <<<<<<< CORRECCIÓN CRÍTICA: weight_decay=0 >>>>>>>
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0) 

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        total_reg_loss = 0
        start_epoch_time = time.time()
        
        # Iterar sobre lotes de muestreo BPR
        for users, pos_items, neg_items in train_loader:
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            
            optimizer.zero_grad()
            
            # Recalcular embeddings finales EN CADA LOTE
            final_user_emb, final_item_emb = model.forward()
            
            # Obtener embeddings iniciales (capa 0)
            initial_user_emb = model.user_embedding.weight
            initial_item_emb = model.item_embedding.weight

            # Calcular pérdidas
            ranking_loss, reg_loss = model.bpr_loss(
                initial_user_emb, initial_item_emb,
                final_user_emb, final_item_emb,
                users, pos_items, neg_items
            )
            
            # Aplicar weight decay manualmente (como en el notebook)
            loss = ranking_loss + reg_loss * WEIGHT_DECAY
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_reg_loss += reg_loss.item() 

        epoch_duration = time.time() - start_epoch_time
        avg_loss = total_loss / len(train_loader)
        avg_reg_loss = total_reg_loss / len(train_loader)
        print(f"   Epoch {epoch+1}/{EPOCHS}, Loss Total: {avg_loss:.4f} (Rank: {avg_loss - avg_reg_loss * WEIGHT_DECAY:.4f}, Reg: {avg_reg_loss * WEIGHT_DECAY:.4f}) (Dur: {epoch_duration:.2f}s)")

    print("   Entrenamiento completado.")
    return model

def generate_predictions(model, prediction_components):
    """Genera predicciones con el modelo LightGCN (Puro PyTorch) entrenado."""
    print("3. Generando predicciones con LightGCN (Puro PyTorch Corregido)...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Usando dispositivo para predicción: {device}")
    model.to(device)
    model.eval()

    antitest_df = prediction_components['antitest_df']
    user_map = prediction_components['user_map']
    item_map = prediction_components['item_map']
    
    # El `edge_index` (norm_adj_matrix) está DENTRO del objeto `model`
    
    print("   Calculando embeddings finales...")
    with torch.no_grad():
        final_user_emb, final_item_emb = model.forward() # Calcular embeddings finales una vez

    antitest_mapped = antitest_df.copy()
    antitest_mapped['user_idx'] = antitest_mapped['userId'].map(user_map)
    antitest_mapped['item_idx'] = antitest_mapped['movieId'].map(item_map)
    valid_antitest_mapped = antitest_mapped.dropna(subset=['user_idx', 'item_idx']).copy()
    valid_antitest_mapped['user_idx'] = valid_antitest_mapped['user_idx'].astype(int)
    valid_antitest_mapped['item_idx'] = valid_antitest_mapped['item_idx'].astype(int)

    # Convertir a tensores para indexar embeddings
    user_indices = torch.from_numpy(valid_antitest_mapped['user_idx'].values).long().to(device)
    item_indices = torch.from_numpy(valid_antitest_mapped['item_idx'].values).long().to(device)

    # Asegurar índices válidos
    num_users_model = final_user_emb.shape[0]
    num_items_model = final_item_emb.shape[0]
    user_mask = user_indices < num_users_model
    item_mask = item_indices < num_items_model
    valid_mask = user_mask & item_mask

    original_count = len(valid_antitest_mapped) # Contar antes de filtrar
    if not valid_mask.all():
         removed_count = torch.sum(~valid_mask).item()
         print(f"   Advertencia: Se descartaron {removed_count} pares con índices fuera de rango.")
         user_indices = user_indices[valid_mask]
         item_indices = item_indices[valid_mask]
         valid_antitest_mapped = valid_antitest_mapped[valid_mask.cpu().numpy()] # Filtrar DF también
    else:
        removed_count = 0

    if len(user_indices) > 0 and len(item_indices) > 0: # Solo si quedan índices válidos
        user_emb_batch = final_user_emb[user_indices]
        item_emb_batch = final_item_emb[item_indices]
    else:
        user_emb_batch, item_emb_batch = None, None # Marcar como None

    # Calcular predicciones solo si hay embeddings válidos
    if user_emb_batch is not None and item_emb_batch is not None:
        
        # <<<<<<< CAMBIO CRÍTICO: Forzar cálculo de predicción en CPU >>>>>>>
        user_emb_batch_cpu = user_emb_batch.to('cpu')
        item_emb_batch_cpu = item_emb_batch.to('cpu')
        
        # Calcular predicciones en CPU
        predictions = torch.einsum("bd,bd->b", user_emb_batch_cpu, item_emb_batch_cpu)
        # predictions = torch.sum(user_emb_batch_cpu * item_emb_batch_cpu, dim=1) # Alternativa

        # Crear DataFrame final
        predictions_df = valid_antitest_mapped.copy() # Usar el DF ya filtrado por valid_mask
        predictions_np = predictions.cpu().numpy() # .cpu() aquí es redundante pero no daña

        predictions_df['prediction'] = predictions_np
        print(f"   Se generaron {len(predictions_df)} predicciones válidas.")

    else: # Si no hubo embeddings válidos
        print("   No se generaron predicciones debido a índices inválidos.")
        # Devolver DF vacío pero con columnas correctas
        predictions_df = pd.DataFrame(columns=['userId', 'movieId', 'prediction'])


    # Siempre devolver las columnas esperadas, aunque esté vacío
    return predictions_df[['userId', 'movieId', 'prediction']]


# Bloque if __name__ == "__main__"
# ... (sin cambios) ...
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python lightgcn_model.py <dataset_percentage>")
        sys.exit(1)

    dataset_percentage = sys.argv[1]
    DATA_PATH = os.path.join('data', dataset_percentage)

    if not os.path.exists(DATA_PATH):
        print(f"Error: El directorio de datos no existe en '{DATA_PATH}'")
        sys.exit(1)

    try:
        train_comps, pred_comps_dict, u_map, i_map = preprocess_data(DATA_PATH)
        # Añadir mapas para la ejecución de prueba local
        pred_comps_dict['user_map'] = u_map
        pred_comps_dict['item_map'] = i_map

        trained_model = train_model(train_comps)
        predictions = generate_predictions(trained_model, pred_comps_dict)

        print("\n--- Proceso del modelo LightGCN finalizado ---")
        if predictions is not None and not predictions.empty:
             print("Ejemplo de 5 predicciones:")
             print(predictions.head())
        else:
             print("No se generaron predicciones.")

    except FileNotFoundError:
        print(f"Error: No se encontraron los archivos train.csv o antitest.csv en '{DATA_PATH}'")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        import traceback
        traceback.print_exc()
