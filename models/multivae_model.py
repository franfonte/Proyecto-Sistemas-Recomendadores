import pandas as pd
import numpy as np
import os
import sys
from scipy.sparse import csr_matrix # Cambiado de coo_matrix a csr_matrix para getrow
import random # Necesario para random.seed
import time # Añadido para medir épocas si es necesario

# Importaciones de PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Arquitectura basada en el paper original (aproximada)
HIDDEN_DIM = 600
LATENT_DIM = 200
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001
BATCH_SIZE = 500
EPOCHS = 30
BETA = 1.0 # Peso del KL divergence (por defecto, no se anea)
WEIGHT_DECAY = 0.0 # Paper original no usa weight decay explícito en Adam


# --- 1. Arquitectura del Modelo MultiVAE ---
class MultiVAE(nn.Module):
    """
    Implementación del modelo Variational Autoencoder for Collaborative Filtering (MultiVAE).
    """
    def __init__(self, num_items, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, dropout=DROPOUT_RATE):
        super(MultiVAE, self).__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim * 2) # Salida mu y logvar
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_items) # Salida reconstruida (logits)
        )
        self.dropout = nn.Dropout(dropout)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) # Determinista si torch.manual_seed está fijado
            return eps.mul(std).add_(mu)
        else:
            return mu # En inferencia, usar la media

    def forward(self, x):
        # Aplicar dropout al input (según paper) solo durante entrenamiento
        if self.training:
            x_dropped = F.dropout(x, p=self.dropout.p, training=self.training) # Usar self.training
        else:
            x_dropped = x # No aplicar dropout durante la evaluación

        # Encoder -> mu, logvar
        h = self.encoder(x_dropped)
        mu, logvar = torch.chunk(h, 2, dim=-1)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decoder -> logits reconstruidos
        logits = self.decoder(z)
        return logits, mu, logvar

# --- 2. Función de Pérdida VAE ---
def vae_loss_function(recon_x_logits, x, mu, logvar, beta=BETA):
    """
    Calcula la pérdida ELBO (Evidence Lower Bound) para el VAE.
    """
    log_softmax_recon = F.log_softmax(recon_x_logits, dim=-1)
    reconstruction_loss = -torch.sum(log_softmax_recon * x, dim=-1).mean()
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    return reconstruction_loss + beta * kl_divergence

# --- 3. Dataset de PyTorch ---
class VAEInteractionDataset(Dataset):
    """
    Dataset para cargar la matriz de interacciones fila por fila (cada fila es un usuario).
    """
    def __init__(self, interactions_matrix_csr):
        self.interactions = interactions_matrix_csr

    def __len__(self):
        return self.interactions.shape[0]

    def __getitem__(self, idx):
        row_sparse = self.interactions[idx]
        row_dense = row_sparse.toarray().squeeze()
        return torch.FloatTensor(row_dense)

# --- 4. Funciones para la Integración con run_experiment.py ---
def preprocess_data(data_path):
    """
    Carga datos y los convierte al formato que MultiVAE necesita.
    """
    print(f"1. Preprocesando datos para MultiVAE desde: {data_path}")
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

    print("   Aplicando regla de binarización: rating >= 4.0 --> 1, < 4.0 se ignora")
    train_df.loc[:, 'rating_bin'] = train_df['rating'].apply(lambda x: 1 if x >= 4.0 else 0)
    train_positive = train_df[train_df['rating_bin'] == 1].copy()

    train_positive.loc[:, 'user_idx'] = train_positive['userId'].map(user_map)
    train_positive.loc[:, 'item_idx'] = train_positive['movieId'].map(item_map)
    train_positive = train_positive.dropna(subset=['user_idx', 'item_idx'])
    train_positive['user_idx'] = train_positive['user_idx'].astype(int)
    train_positive['item_idx'] = train_positive['item_idx'].astype(int)
    print(f"   Interacciones positivas (>=4) a usar para entrenamiento: {len(train_positive)}")

    interactions_matrix = csr_matrix(
        (np.ones(len(train_positive)), (train_positive['user_idx'], train_positive['item_idx'])),
        shape=(num_users, num_items),
        dtype=np.float32
    )
    print("   Matriz de interacciones CSR creada.")

    train_dataset = VAEInteractionDataset(interactions_matrix)
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g
    )

    training_components = {'train_loader': train_loader, 'num_items': num_items}
    prediction_components = {
        'interactions_matrix': interactions_matrix,
        'antitest_df': antitest_df,
        'user_map': user_map,
        'item_map': item_map
    }
    return training_components, prediction_components

def train_model(training_components):
    """
    Entrena el modelo MultiVAE.
    """
    set_seed(RANDOM_SEED)
    print("2. Entrenando el modelo MultiVAE...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Usando dispositivo: {device}")

    train_loader = training_components['train_loader']
    num_items = training_components['num_items']

    model = MultiVAE(num_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    model.train()
    total_anneal_steps = 200000
    update_count = 0

    for epoch in range(EPOCHS):
        total_loss = 0
        start_epoch_time = time.time()
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            recon_logits, mu, logvar = model(batch_data) # Llamar a forward

            if total_anneal_steps > 0:
                anneal = min(BETA, update_count / total_anneal_steps)
            else:
                anneal = BETA
            update_count += 1

            loss = vae_loss_function(recon_logits, batch_data, mu, logvar, beta=anneal)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_duration = time.time() - start_epoch_time
        avg_loss = total_loss / len(train_loader)
        print(f"   Epoch {epoch+1}/{EPOCHS}, ELBO Loss: {avg_loss:.4f} (Anneal: {anneal:.4f}, Duración: {epoch_duration:.2f}s)")

    print("   Entrenamiento completado.")
    return model

def generate_predictions(model, prediction_components):
    """
    Genera predicciones para el conjunto antitest con MultiVAE.
    """
    print("3. Generando predicciones con MultiVAE...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Usando dispositivo para predicción: {device}")
    model.to(device)
    model.eval() # Poner en modo evaluación

    interactions_matrix = prediction_components['interactions_matrix']
    antitest_df = prediction_components['antitest_df']
    user_map = prediction_components['user_map']
    item_map = prediction_components['item_map']

    # Mapear antitest
    antitest_mapped = antitest_df.copy()
    antitest_mapped['user_idx'] = antitest_mapped['userId'].map(user_map)
    antitest_mapped['item_idx'] = antitest_mapped['movieId'].map(item_map)
    valid_antitest_mapped = antitest_mapped.dropna(subset=['user_idx', 'item_idx']).copy()
    valid_antitest_mapped['user_idx'] = valid_antitest_mapped['user_idx'].astype(int)
    valid_antitest_mapped['item_idx'] = valid_antitest_mapped['item_idx'].astype(int)

    # Preparar DataLoader para predicción
    unique_user_indices = valid_antitest_mapped['user_idx'].unique()
    # Asegurar que los índices estén ordenados para mapeo correcto después
    unique_user_indices.sort()
    pred_dataset = VAEInteractionDataset(interactions_matrix[unique_user_indices])
    pred_loader = DataLoader(pred_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    # Diccionario para almacenar las reconstrucciones por user_idx
    user_recon_scores = {}

    with torch.no_grad():
        current_idx_in_batch = 0 # Índice dentro del loader
        for batch_data in pred_loader:
            batch_data = batch_data.to(device)
            recon_logits, mu, _ = model(batch_data)
            log_probs = F.log_softmax(recon_logits, dim=-1)

            batch_size = batch_data.shape[0]
            # Mapear los scores de vuelta a los user_idx originales
            original_indices = unique_user_indices[current_idx_in_batch : current_idx_in_batch + batch_size]
            for i, original_user_idx in enumerate(original_indices):
                user_recon_scores[original_user_idx] = log_probs[i].cpu().numpy()
            current_idx_in_batch += batch_size


    # Extraer los scores específicos para los pares (user_idx, item_idx) del antitest
    scores = []
    # Usar .apply() puede ser más eficiente que iterrows() para DataFrames grandes
    def get_score(row):
        user_idx = row['user_idx']
        item_idx = row['item_idx']
        if user_idx in user_recon_scores:
            # <<<<<<< CAMBIO AQUÍ: Convertir item_idx a int explícitamente >>>>>>>
            try:
                # Asegurarse de que item_idx sea un entero válido
                item_idx_int = int(item_idx)
                # Verificar límites (opcional pero seguro)
                if 0 <= item_idx_int < len(user_recon_scores[user_idx]):
                     return user_recon_scores[user_idx][item_idx_int]
                else:
                     print(f"   Advertencia: item_idx {item_idx_int} fuera de rango para user_idx {user_idx}. Longitud: {len(user_recon_scores[user_idx])}")
                     return np.nan
            except ValueError:
                 print(f"   Advertencia: No se pudo convertir item_idx {item_idx} a int para user_idx {user_idx}.")
                 return np.nan # Manejar error de conversión
        else:
             return np.nan # Usuario no encontrado (no debería pasar)

    predictions_df = valid_antitest_mapped.copy()
    predictions_df['prediction'] = predictions_df.apply(get_score, axis=1)


    # Eliminar filas donde no se pudo generar score (si hubo NaNs)
    rows_before = len(predictions_df)
    predictions_df = predictions_df.dropna(subset=['prediction'])
    rows_after = len(predictions_df)
    if rows_after < rows_before:
        print(f"   Advertencia: Se descartaron {rows_before - rows_after} filas debido a scores NaN.")


    print(f"   Se generaron {len(predictions_df)} predicciones.")
    return predictions_df[['userId', 'movieId', 'prediction']]