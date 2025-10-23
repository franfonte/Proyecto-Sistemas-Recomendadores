import pandas as pd
import os
import numpy as np
import random

# Importaciones de PyTorch y PyTorch Geometric
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp

# --- Hiperparámetros por Defecto ---
EMBEDDING_DIM = 64
NUM_LAYERS = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 1024 # Aumentado para un entrenamiento más estable y rápido
EPOCHS = 30 # Usamos 30 como un buen punto de partida
WEIGHT_DECAY = 1e-4 # Parámetro de regularización L2

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
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.norm_adj_matrix = norm_adj_matrix.coalesce()

    def forward(self):
        initial_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_layer_embeddings = [initial_embeddings]
        current_embeddings = initial_embeddings

        for _ in range(self.num_layers):
            cpu_embeddings = current_embeddings.to('cpu')
            propagated_embeddings = torch.sparse.mm(self.norm_adj_matrix, cpu_embeddings)
            current_embeddings = propagated_embeddings.to(initial_embeddings.device)
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

        # <<<<<<< CAMBIO CRÍTICO: Se eliminó el término de regularización manual >>>>>>>
        # La regularización L2 ya está siendo manejada por el `weight_decay` en el optimizador Adam.
        # Mantenerlo aquí era redundante y probablemente causaba el error de aprendizaje.
        
        return loss

# --- 2. Dataset y Funciones Auxiliares ---
class BPRDataset(Dataset):
    def __init__(self, train_df, num_items, user_interactions):
        self.users = torch.tensor(train_df['user_idx'].values, dtype=torch.long)
        self.pos_items = torch.tensor(train_df['item_idx'].values, dtype=torch.long)
        self.num_items = num_items
        self.user_interactions = user_interactions

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx].item()
        pos_item = self.pos_items[idx].item()
        
        neg_item = random.randint(0, self.num_items - 1)
        while neg_item in self.user_interactions.get(user, []):
            neg_item = random.randint(0, self.num_items - 1)
            
        return user, pos_item, neg_item

def create_norm_adj_matrix(edge_index, num_users, num_items):
    user_nodes = edge_index[0].numpy().copy()
    item_nodes = edge_index[1].numpy().copy()

    adj_matrix = sp.dok_matrix((num_users + num_items, num_users + num_items), dtype=np.float32)
    adj_matrix = adj_matrix.tolil()
    
    item_nodes_shifted = item_nodes + num_users
    adj_matrix[user_nodes, item_nodes_shifted] = 1
    adj_matrix[item_nodes_shifted, user_nodes] = 1
    adj_matrix = adj_matrix.todok()

    row_sum = np.array(adj_matrix.sum(axis=1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    norm_adj_matrix = d_mat_inv_sqrt.dot(adj_matrix).dot(d_mat_inv_sqrt).tocoo()

    row = torch.from_numpy(norm_adj_matrix.row.astype(np.int64))
    col = torch.from_numpy(norm_adj_matrix.col.astype(np.int64))
    val = torch.from_numpy(norm_adj_matrix.data.astype(np.float32))
    
    return torch.sparse_coo_tensor(torch.stack([row, col]), val, norm_adj_matrix.shape)

# --- 3. Funciones para la Integración con run_experiment.py ---
def preprocess_data(data_path):
    print(f"1. Preprocesando datos para LightGCN desde: {data_path}")
    train_file = os.path.join(data_path, 'train.csv')
    antitest_file = os.path.join(data_path, 'antitest.csv')
    train_df = pd.read_csv(train_file)
    antitest_df = pd.read_csv(antitest_file)

    user_ids = pd.concat([train_df['userId'], antitest_df['userId']]).unique()
    item_ids = pd.concat([train_df['movieId'], antitest_df['movieId']]).unique()
    user_map = {uid: i for i, uid in enumerate(user_ids)}
    item_map = {iid: i for i, iid in enumerate(item_ids)}
    num_users, num_items = len(user_map), len(item_map)

    train_df['user_idx'] = train_df['userId'].map(user_map)
    train_df['item_idx'] = train_df['movieId'].map(item_map)
    
    user_interactions = train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()

    edge_index_np = np.vstack([train_df['user_idx'].values, train_df['item_idx'].values])
    edge_index = torch.from_numpy(edge_index_np).long()
    
    norm_adj_matrix = create_norm_adj_matrix(edge_index, num_users, num_items)
    
    train_dataset = BPRDataset(train_df, num_items, user_interactions)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    training_components = {
        'train_loader': train_loader,
        'num_users': num_users,
        'num_items': num_items,
        'norm_adj_matrix': norm_adj_matrix
    }
    prediction_components = {
        'antitest_df': antitest_df,
        'user_map': user_map,
        'item_map': item_map
    }
    return training_components, prediction_components

def train_model(training_components):
    print("2. Entrenando el modelo LightGCN...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Usando dispositivo: {device}")

    train_loader = training_components['train_loader']
    num_users = training_components['num_users']
    num_items = training_components['num_items']
    norm_adj_matrix = training_components['norm_adj_matrix']

    model = LightGCN(num_users, num_items, NUM_LAYERS, EMBEDDING_DIM, norm_adj_matrix).to(device)
    # Aquí, `weight_decay` se encarga de la regularización L2 por nosotros.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for users, pos_items, neg_items in train_loader:
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            
            optimizer.zero_grad()
            loss = model.bpr_loss(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"   Epoch {epoch+1}/{EPOCHS}, BPR Loss: {total_loss/len(train_loader):.4f}")

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
    
    antitest_df['user_idx'] = antitest_df['userId'].map(user_map)
    antitest_df['item_idx'] = antitest_df['movieId'].map(item_map)
    valid_antitest = antitest_df.dropna(subset=['user_idx', 'item_idx']).copy()
    valid_antitest['user_idx'] = valid_antitest['user_idx'].astype(int)
    valid_antitest['item_idx'] = valid_antitest['item_idx'].astype(int)
    
    user_indices = torch.tensor(valid_antitest['user_idx'].values).to(device)
    item_indices = torch.tensor(valid_antitest['item_idx'].values).to(device)
    
    user_emb_batch = final_user_emb[user_indices]
    item_emb_batch = final_item_emb[item_indices]
    
    predictions = torch.sum(user_emb_batch * item_emb_batch, dim=1)
    
    predictions_df = valid_antitest.copy()
    predictions_df['prediction'] = predictions.cpu().numpy()
    
    print(f"   Se generaron {len(predictions_df)} predicciones.")
    return predictions_df[['userId', 'movieId', 'prediction']]