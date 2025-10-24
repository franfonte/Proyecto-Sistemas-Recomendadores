import pandas as pd
import os
import sys
import numpy as np
import time
import random # Necesario para random.seed

# Importaciones de PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Constante para Replicabilidad ---
RANDOM_SEED = 42

# --- Función para Fijar Semillas ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.mps.manual_seed(seed) # Descomentar si se usa MPS y se requiere
    print(f"   Semillas fijadas en: {seed}")

# --- Worker Init Fn para DataLoader ---
def seed_worker(worker_id):
    # Asegura que cada worker del DataLoader tenga una semilla reproducible pero diferente
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --- Hiperparámetros por Defecto ---
EMBEDDING_DIM = 64
MLP_LAYERS = [64, 32, 16]
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 10

# --- 1. Definición del Dataset para PyTorch ---
class MovieLensDataset(Dataset):
    """
    Dataset personalizado de PyTorch para cargar los datos de ratings.
    """
    def __init__(self, users, items, ratings=None):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float) if ratings is not None else None

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        if self.ratings is not None:
            return self.users[idx], self.items[idx], self.ratings[idx]
        return self.users[idx], self.items[idx]

# --- 2. Arquitectura del Modelo NCF ---
class NCF(nn.Module):
    """
    Implementación del modelo Neural Collaborative Filtering (NCF).
    Combina Generalized Matrix Factorization (GMF) y un Multi-Layer Perceptron (MLP).
    """
    def __init__(self, num_users, num_items, embedding_dim, mlp_layers):
        super(NCF, self).__init__()

        # --- Capas de Embedding ---
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)

        # --- Capas del MLP ---
        self.mlp_layers = nn.ModuleList()
        input_size = embedding_dim * 2
        for layer_size in mlp_layers:
            self.mlp_layers.append(nn.Linear(input_size, layer_size))
            input_size = layer_size

        # --- Capa de Predicción Final ---
        predict_input_size = embedding_dim + mlp_layers[-1]
        self.predict_layer = nn.Linear(predict_input_size, 1)

        # Inicialización explícita controlada por la semilla global
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.gmf_user_embedding.weight)
        nn.init.xavier_uniform_(self.gmf_item_embedding.weight)
        nn.init.xavier_uniform_(self.mlp_user_embedding.weight)
        nn.init.xavier_uniform_(self.mlp_item_embedding.weight)
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias) # Inicializar bias a cero es común
        nn.init.xavier_uniform_(self.predict_layer.weight)
        nn.init.zeros_(self.predict_layer.bias)


    def forward(self, user_indices, item_indices):
        gmf_user_emb = self.gmf_user_embedding(user_indices)
        gmf_item_emb = self.gmf_item_embedding(item_indices)
        gmf_output = gmf_user_emb * gmf_item_emb

        mlp_user_emb = self.mlp_user_embedding(user_indices)
        mlp_item_emb = self.mlp_item_embedding(item_indices)
        mlp_input = torch.cat([mlp_user_emb, mlp_item_emb], dim=-1)

        mlp_output = mlp_input
        for layer in self.mlp_layers:
            mlp_output = torch.relu(layer(mlp_output))

        concat_output = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = self.predict_layer(concat_output)

        return prediction.squeeze()

# --- 3. Funciones para la Integración con run_experiment.py ---
def preprocess_data(data_path):
    """
    Carga y preprocesa los datos para el modelo NCF.
    Crea mapeos de IDs, y prepara DataLoaders de PyTorch.
    """
    print(f"1. Preprocesando datos para NCF desde: {data_path}")
    train_file = os.path.join(data_path, 'train.csv')
    antitest_file = os.path.join(data_path, 'antitest.csv')

    train_df = pd.read_csv(train_file)
    antitest_df = pd.read_csv(antitest_file)

    user_ids = pd.concat([train_df['userId'], antitest_df['userId']]).unique()
    item_ids = pd.concat([train_df['movieId'], antitest_df['movieId']]).unique()

    user_map = {uid: i for i, uid in enumerate(user_ids)}
    item_map = {iid: i for i, iid in enumerate(item_ids)}

    num_users = len(user_map)
    num_items = len(item_map)
    print(f"   Usuarios únicos totales (train+antitest): {num_users}")
    print(f"   Items únicos totales (train+antitest): {num_items}")


    train_df['user_idx'] = train_df['userId'].map(user_map)
    train_df['item_idx'] = train_df['movieId'].map(item_map)
    # Asegurar que no haya NaNs después del mapeo y convertir a int
    train_df = train_df.dropna(subset=['user_idx', 'item_idx'])
    train_df['user_idx'] = train_df['user_idx'].astype(int)
    train_df['item_idx'] = train_df['item_idx'].astype(int)


    dataset = MovieLensDataset(
        train_df['user_idx'].values,
        train_df['item_idx'].values,
        train_df['rating'].values
    )
    # <<<<<<< CAMBIO AQUÍ: Añadido worker_init_fn y generator >>>>>>>
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0, # Fijar a 0 para máxima replicabilidad inicial
        worker_init_fn=seed_worker,
        generator=g
    )

    print("   DataLoaders y mapeos creados.")

    training_components = {
        'train_loader': train_loader,
        'num_users': num_users,
        'num_items': num_items
    }

    prediction_components = {
        'antitest_df': antitest_df,
        'user_map': user_map,
        'item_map': item_map
    }

    return training_components, prediction_components

def train_model(training_components):
    """
    Entrena el modelo NCF, usando la GPU (MPS) si está disponible.
    """
    # <<<<<<< CAMBIO AQUÍ: Fijar semillas al inicio >>>>>>>
    set_seed(RANDOM_SEED)

    print("2. Entrenando el modelo NCF...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Usando dispositivo: {device}")

    train_loader = training_components['train_loader']
    num_users = training_components['num_users']
    num_items = training_components['num_items']

    model = NCF(num_users, num_items, EMBEDDING_DIM, MLP_LAYERS).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for users, items, ratings in train_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)

            optimizer.zero_grad()
            predictions = model(users, items)
            loss = loss_function(predictions, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"   Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

    print("   Entrenamiento completado.")
    return model

def generate_predictions(model, prediction_components):
    """
    Genera predicciones, usando la GPU (MPS) si está disponible.
    """
    print("3. Generando predicciones con NCF...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Usando dispositivo para predicción: {device}")

    antitest_df = prediction_components['antitest_df']
    user_map = prediction_components['user_map']
    item_map = prediction_components['item_map']

    antitest_mapped = antitest_df.copy()
    antitest_mapped['user_idx'] = antitest_mapped['userId'].map(user_map)
    antitest_mapped['item_idx'] = antitest_mapped['movieId'].map(item_map)

    valid_antitest_mapped = antitest_mapped.dropna(subset=['user_idx', 'item_idx']).copy()
    valid_antitest_mapped['user_idx'] = valid_antitest_mapped['user_idx'].astype(int)
    valid_antitest_mapped['item_idx'] = valid_antitest_mapped['item_idx'].astype(int)

    pred_dataset = MovieLensDataset(
        valid_antitest_mapped['user_idx'].values,
        valid_antitest_mapped['item_idx'].values
    )
    # Usar un batch size mayor en predicción es más eficiente
    pred_loader = DataLoader(pred_dataset, batch_size=BATCH_SIZE * 4, shuffle=False)

    model.to(device)
    model.eval()

    all_preds = []
    total_batches = len(pred_loader)
    start_time = time.time()

    with torch.no_grad():
        for i, (users, items) in enumerate(pred_loader):
            users, items = users.to(device), items.to(device)

            predictions = model(users, items)
            all_preds.extend(predictions.cpu().numpy().tolist())

            # Quitado el print de progreso para reducir output, ya que no es interactivo
            # if (i + 1) % 100 == 0:
            #     elapsed = time.time() - start_time
            #     print(f"   Procesado lote {i+1}/{total_batches} ({elapsed:.2f}s transcurridos)")

    predictions_df = valid_antitest_mapped.copy()
    # Asegurarse de que el número de predicciones coincida
    if len(all_preds) == len(predictions_df):
        predictions_df['prediction'] = all_preds
    else:
         print(f"   [Error] Mismatch en longitud de predicciones: {len(all_preds)} vs {len(predictions_df)}")
         # Devolver DF sin columna 'prediction' en caso de error
         return predictions_df[['userId', 'movieId']]


    print(f"   Se generaron {len(predictions_df)} predicciones.")
    return predictions_df[['userId', 'movieId', 'prediction']]