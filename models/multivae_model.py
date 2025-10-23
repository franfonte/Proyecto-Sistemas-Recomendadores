import pandas as pd
import os
import numpy as np
from scipy.sparse import coo_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Hiperparámetros por Defecto ---
LATENT_DIM = 64
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 30
WEIGHT_DECAY = 0.01

# --- 1. Arquitectura del Modelo MultiVAE ---
class MultiVAE(nn.Module):
    """
    Implementación de Variational Autoencoder for Collaborative Filtering (MultiVAE).
    """
    def __init__(self, num_items, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM):
        super(MultiVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim * 2) # Dos cabezas: media (mu) y log-varianza (logvar)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_items)
            # La salida pasará por un LogSoftmax en la pérdida
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar, anneal=1.0):
    """
    Pérdida para el VAE. Combina la Reconstrucción (Multinomial Logistic) y la Regularización KL.
    """
    # Pérdida de Reconstrucción (Log-verosimilitud Multinomial)
    recon_loss = -torch.mean(torch.sum(torch.log_softmax(recon_x, 1) * x, dim=-1))
    
    # Pérdida de Regularización KL (Divergencia KL)
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    return recon_loss + anneal * kl_loss

# --- 2. Dataset y Funciones Auxiliares ---
class VAEDataLoader(Dataset):
    """
    Dataset para cargar los datos de interacciones dispersas de usuarios.
    """
    def __init__(self, interactions_matrix):
        # Convertir a formato LIL para una indexación de filas más rápida
        self.matrix = interactions_matrix.tolil()

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, idx):
        # Obtener la fila (historial de interacciones del usuario)
        row = self.matrix.getrow(idx).toarray().squeeze()
        return torch.FloatTensor(row)

# --- 3. Funciones para la Integración con run_experiment.py ---
def preprocess_data(data_path):
    """
    Carga y preprocesa los datos para MultiVAE.
    Crea una matriz dispersa de interacciones usuario-ítem.
    """
    print(f"1. Preprocesando datos para MultiVAE desde: {data_path}")
    train_file = os.path.join(data_path, 'train.csv')
    antitest_file = os.path.join(data_path, 'antitest.csv')
    
    train_df = pd.read_csv(train_file)
    antitest_df = pd.read_csv(antitest_file)

    # Combinar todos los datos para crear mapeos consistentes
    all_df = pd.concat([train_df[['userId', 'movieId']], antitest_df[['userId', 'movieId']]])
    user_ids = all_df['userId'].unique()
    item_ids = all_df['movieId'].unique()
    
    user_map = {uid: i for i, uid in enumerate(user_ids)}
    item_map = {iid: i for i, iid in enumerate(item_ids)}
    
    num_users = len(user_map)
    num_items = len(item_map)

    # --- Aplicar la nueva lógica de Binarización (>= 4.0) ---
    print(f"   Aplicando umbral de binarización: ratings >= 4.0")
    
    # Binarizar: 1 si el rating es >= 4.0, 0 si es < 4.0
    train_df['rating'] = (train_df['rating'] >= 4.0).astype(int)
    
    # Filtrar solo las interacciones positivas (rating == 1)
    train_df = train_df[train_df['rating'] > 0]
    # --- Fin de la nueva lógica ---

    # Mapear IDs a índices
    train_df['user_idx'] = train_df['userId'].map(user_map)
    train_df['item_idx'] = train_df['movieId'].map(item_map)
    
    # Crear la matriz de interacciones dispersa
    interactions = coo_matrix((np.ones(train_df.shape[0]), 
                               (train_df['user_idx'], train_df['item_idx'])),
                              shape=(num_users, num_items),
                              dtype=np.float32)
    
    interactions_csr = interactions.tocsr()
    
    # Crear el DataLoader
    dataset = VAEDataLoader(interactions_csr)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print("   Matriz dispersa y DataLoader creados.")
    
    training_components = {
        'train_loader': train_loader,
        'num_items': num_items
    }
    
    prediction_components = {
        'antitest_df': antitest_df,
        'user_map': user_map,
        'item_map': item_map,
        'interactions_csr': interactions_csr # Se necesita para obtener el historial del usuario
    }
    
    return training_components, prediction_components

def train_model(training_components):
    """
    Entrena el modelo MultiVAE.
    """
    print("2. Entrenando el modelo MultiVAE...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Usando dispositivo: {device}")

    train_loader = training_components['train_loader']
    num_items = training_components['num_items']

    model = MultiVAE(num_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"   Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

    print("   Entrenamiento completado.")
    return model

def generate_predictions(model, prediction_components):
    """
    Genera predicciones para el conjunto antitest usando el modelo MultiVAE entrenado.
    """
    print("3. Generando predicciones con MultiVAE...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Usando dispositivo para predicción: {device}")
    model.to(device)
    model.eval()

    antitest_df = prediction_components['antitest_df']
    user_map = prediction_components['user_map']
    item_map = prediction_components['item_map']
    interactions_csr = prediction_components['interactions_csr']
    
    # Mapear IDs de antitest. Ignorar usuarios/ítems no vistos
    antitest_df['user_idx'] = antitest_df['userId'].map(user_map)
    antitest_df['item_idx'] = antitest_df['movieId'].map(item_map)
    valid_antitest = antitest_df.dropna(subset=['user_idx', 'item_idx']).copy()
    valid_antitest['user_idx'] = valid_antitest['user_idx'].astype(int)
    valid_antitest['item_idx'] = valid_antitest['item_idx'].astype(int)

    all_predictions = []
    
    with torch.no_grad():
        for user_idx in valid_antitest['user_idx'].unique():
            # Obtener el historial de interacciones de este usuario
            user_history = torch.FloatTensor(interactions_csr.getrow(user_idx).toarray()).to(device)
            
            # Pasar el historial por el modelo para obtener las reconstrucciones (predicciones)
            # Solo necesitamos la media (mu) para la predicción, no el ruido
            mu, _ = model.encode(user_history)
            recon_scores = model.decoder(mu)
            
            # Aplicar LogSoftmax para obtener probabilidades
            recon_scores = torch.log_softmax(recon_scores, dim=1)
            recon_scores = recon_scores.squeeze().cpu().numpy()
            
            # Obtener los ítems que necesitamos predecir para este usuario
            user_antitest_items = valid_antitest[valid_antitest['user_idx'] == user_idx]
            item_indices = user_antitest_items['item_idx'].values
            
            # Extraer las puntuaciones solo para los ítems del antitest
            scores = recon_scores[item_indices]
            
            preds_df = pd.DataFrame({
                'userId': user_antitest_items['userId'],
                'movieId': user_antitest_items['movieId'],
                'prediction': scores
            })
            all_predictions.append(preds_df)

    if not all_predictions:
        print("   No se generaron predicciones válidas.")
        return pd.DataFrame(columns=['userId', 'movieId', 'prediction'])

    predictions_df = pd.concat(all_predictions)
    print(f"   Se generaron {len(predictions_df)} predicciones.")
    return predictions_df[['userId', 'movieId', 'prediction']]