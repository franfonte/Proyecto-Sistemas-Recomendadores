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

# --- Constante para Replicabilidad ---
RANDOM_SEED = 42

# --- Función para Fijar Semillas ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.mps.manual_seed(seed) # Descomentar si se usa MPS y se requiere
    # Para mayor determinismo en CUDA (no aplica a MPS directamente, pero buena práctica si se cambia de backend)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"   Semillas fijadas en: {seed}")

# --- Worker Init Fn para DataLoader ---
def seed_worker(worker_id):
    # Asegura que cada worker del DataLoader tenga una semilla reproducible pero diferente
    # Es crucial para el muestreo negativo en BPRDataset
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --- Hiperparámetros por Defecto ---
EMBEDDING_DIM = 64
NUM_LAYERS = 3
LEARNING_RATE = 0.0001
BATCH_SIZE = 1024
EPOCHS = 30
WEIGHT_DECAY = 1e-4 # Parámetro de regularización L2 estándar

# --- 1. Arquitectura del Modelo LightGCN ---
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, num_layers, embedding_dim, norm_adj_matrix):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        # La inicialización se controla ahora con torch.manual_seed global
        # nn.init.xavier_uniform_(self.user_embedding.weight)
        # nn.init.xavier_uniform_(self.item_embedding.weight)

        # La conversión y coalesce se hacen al recibir la matriz
        if isinstance(norm_adj_matrix, torch.Tensor):
             # Asegurarse de que el tensor sparse esté coalesced
             self.norm_adj_matrix = norm_adj_matrix.coalesce()
        else:
             # Si viene de scipy, convertirla a tensor sparse de PyTorch y coalesced
             self.norm_adj_matrix = self.convert_sp_mat_to_sp_tensor(norm_adj_matrix).coalesce()


    # Helper function para convertir matriz dispersa de SciPy a Tensor disperso de PyTorch
    def convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        # Usar torch.from_numpy para evitar copia innecesaria si es posible
        row = torch.from_numpy(coo.row).long()
        col = torch.from_numpy(coo.col).long()
        index = torch.stack([row, col])
        data = torch.from_numpy(coo.data).float()
        # Usar la función moderna recomendada
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

    def forward(self):
        initial_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_layer_embeddings = [initial_embeddings]
        current_embeddings = initial_embeddings
        device = initial_embeddings.device

        for _ in range(self.num_layers):
            # Forzar CPU para la multiplicación dispersa si MPS no la soporta
            cpu_embeddings = current_embeddings.to('cpu')
            # Asegurarse de que la matriz esté en CPU también
            norm_adj_matrix_cpu = self.norm_adj_matrix.to('cpu')
            propagated_embeddings = torch.sparse.mm(norm_adj_matrix_cpu, cpu_embeddings)
            # Mover resultado de vuelta al dispositivo original
            current_embeddings = propagated_embeddings.to(device)
            all_layer_embeddings.append(current_embeddings)

        final_embeddings = torch.mean(torch.stack(all_layer_embeddings, dim=1), dim=1)
        final_user_emb, final_item_emb = torch.split(final_embeddings, [self.num_users, self.num_items])
        return final_user_emb, final_item_emb

    def bpr_loss(self, users, pos_items, neg_items):
        final_user_emb, final_item_emb = self.forward()
        user_emb = final_user_emb[users]
        pos_item_emb = final_item_emb[pos_items]
        neg_item_emb = final_item_emb[neg_items]
        score_diff = torch.sum(user_emb * (pos_item_emb - neg_item_emb), dim=1)
        loss = -torch.mean(torch.log(torch.sigmoid(score_diff) + 1e-9))
        return loss

# --- 2. Dataset y Funciones Auxiliares ---
class BPRDataset(Dataset):
    def __init__(self, train_df, num_items, user_interactions):
        self.users = torch.tensor(train_df['user_idx'].values, dtype=torch.long)
        self.pos_items = torch.tensor(train_df['item_idx'].values, dtype=torch.long)
        self.num_items = num_items
        self.user_interactions = user_interactions # Diccionario {user_idx: {item_idx1, item_idx2, ...}}

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx].item()
        pos_item = self.pos_items[idx].item()
        user_seen_items = self.user_interactions.get(user, set())

        # El muestreo negativo ahora será determinista gracias a seed_worker y el generador del DataLoader
        neg_item = random.randint(0, self.num_items - 1)
        # Bucle para asegurar que sea un ítem no visto
        # En teoría, con suficientes items, esto no debería tardar mucho
        # Considerar muestreo más eficiente si num_items es muy grande comparado con items vistos
        while neg_item in user_seen_items:
             neg_item = random.randint(0, self.num_items - 1)

        return user, pos_item, neg_item

def create_norm_adj_matrix(edge_index_np, num_users, num_items):
    """
    Crea la matriz de adyacencia normalizada simétricamente para LightGCN.
    Devuelve una matriz dispersa de SciPy (no un tensor de PyTorch).
    """
    user_nodes = edge_index_np[0]
    item_nodes = edge_index_np[1]
    rows = np.concatenate([user_nodes, item_nodes + num_users])
    cols = np.concatenate([item_nodes + num_users, user_nodes])
    data = np.ones(len(rows))
    adj_matrix = sp.coo_matrix((data, (rows, cols)), shape=(num_users + num_items, num_users + num_items), dtype=np.float32)
    row_sum = np.array(adj_matrix.sum(axis=1)).flatten()
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj_matrix = d_mat_inv_sqrt.dot(adj_matrix).dot(d_mat_inv_sqrt)
    print("   Matriz de adyacencia normalizada creada (formato SciPy COO).")
    return norm_adj_matrix

# --- 3. Funciones para la Integración con run_experiment.py ---
def preprocess_data(data_path):
    print(f"1. Preprocesando datos para LightGCN desde: {data_path}")
    train_file = os.path.join(data_path, 'train.csv')
    antitest_file = os.path.join(data_path, 'antitest.csv')
    train_df = pd.read_csv(train_file)
    antitest_df = pd.read_csv(antitest_file)

    all_users = pd.concat([train_df['userId'], antitest_df['userId']]).unique()
    all_items = pd.concat([train_df['movieId'], antitest_df['movieId']]).unique()
    user_map = {uid: i for i, uid in enumerate(all_users)}
    item_map = {iid: i for i, iid in enumerate(all_items)}
    num_users, num_items = len(user_map), len(item_map)
    print(f"   Usuarios únicos totales (train+antitest): {num_users}")
    print(f"   Items únicos totales (train+antitest): {num_items}")

    train_df['user_idx'] = train_df['userId'].map(user_map)
    train_df['item_idx'] = train_df['movieId'].map(item_map)
    train_df = train_df.dropna(subset=['user_idx', 'item_idx'])
    train_df['user_idx'] = train_df['user_idx'].astype(int)
    train_df['item_idx'] = train_df['item_idx'].astype(int)

    user_interactions = train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
    edge_index_np = np.vstack([train_df['user_idx'].values, train_df['item_idx'].values])
    norm_adj_matrix_scipy = create_norm_adj_matrix(edge_index_np, num_users, num_items)

    train_dataset = BPRDataset(train_df, num_items, user_interactions)
    # <<<<<<< CAMBIO AQUÍ: Añadido worker_init_fn y generator >>>>>>>
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0, # Fijar a 0 para máxima replicabilidad inicial
        worker_init_fn=seed_worker,
        generator=g
    )

    training_components = {
        'train_loader': train_loader,
        'num_users': num_users,
        'num_items': num_items,
        'norm_adj_matrix': norm_adj_matrix_scipy
    }
    prediction_components = {
        'antitest_df': antitest_df,
        'user_map': user_map,
        'item_map': item_map
    }
    return training_components, prediction_components

def train_model(training_components):
    """
    Entrena el modelo LightGCN.
    """
    # <<<<<<< CAMBIO AQUÍ: Fijar semillas al inicio >>>>>>>
    set_seed(RANDOM_SEED)

    print("2. Entrenando el modelo LightGCN...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Usando dispositivo: {device}")

    train_loader = training_components['train_loader']
    num_users = training_components['num_users']
    num_items = training_components['num_items']
    norm_adj_matrix_scipy = training_components['norm_adj_matrix']

    model = LightGCN(num_users, num_items, NUM_LAYERS, EMBEDDING_DIM, norm_adj_matrix_scipy).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        start_epoch_time = time.time()

        for users, pos_items, neg_items in train_loader:
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            optimizer.zero_grad()
            loss = model.bpr_loss(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_duration = time.time() - start_epoch_time
        avg_loss = total_loss / len(train_loader)
        print(f"   Epoch {epoch+1}/{EPOCHS}, BPR Loss: {avg_loss:.4f} (Duración: {epoch_duration:.2f}s)")

    print("   Entrenamiento completado.")
    return model

def generate_predictions(model, prediction_components):
    print("3. Generando predicciones con LightGCN...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Usando dispositivo para predicción: {device}")
    model.to(device)
    model.eval()

    antitest_df = prediction_components['antitest_df']
    user_map = prediction_components['user_map']
    item_map = prediction_components['item_map']

    with torch.no_grad():
        final_user_emb, final_item_emb = model.forward()

    antitest_mapped = antitest_df.copy()
    antitest_mapped['user_idx'] = antitest_mapped['userId'].map(user_map)
    antitest_mapped['item_idx'] = antitest_mapped['movieId'].map(item_map)
    valid_antitest_mapped = antitest_mapped.dropna(subset=['user_idx', 'item_idx']).copy()
    valid_antitest_mapped['user_idx'] = valid_antitest_mapped['user_idx'].astype(int)
    valid_antitest_mapped['item_idx'] = valid_antitest_mapped['item_idx'].astype(int)

    # Convertir a tensores para indexar embeddings
    # Usar torch.from_numpy puede ser marginalmente más rápido si no hay copia
    user_indices = torch.from_numpy(valid_antitest_mapped['user_idx'].values).long().to(device)
    item_indices = torch.from_numpy(valid_antitest_mapped['item_idx'].values).long().to(device)

    user_emb_batch = final_user_emb[user_indices]
    item_emb_batch = final_item_emb[item_indices]

    predictions = torch.sum(user_emb_batch * item_emb_batch, dim=1)

    predictions_df = valid_antitest_mapped.copy()
    predictions_df['prediction'] = predictions.cpu().numpy()

    print(f"   Se generaron {len(predictions_df)} predicciones.")
    return predictions_df[['userId', 'movieId', 'prediction']]