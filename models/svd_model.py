#!/usr/bin/env python3
"""
Sample SVD (Singular Value Decomposition) model for recommender systems.
This script simulates a simple recommendation model for demonstration purposes.
"""
import time
import numpy as np
import pandas as pd


def generate_sample_data(n_users=1000, n_items=500):
    """Generate sample rating data for demonstration."""
    print("Generating sample rating data...")
    np.random.seed(42)
    
    # Generate sparse ratings (not all users rate all items)
    n_ratings = int(n_users * n_items * 0.1)  # 10% density
    user_ids = np.random.randint(0, n_users, n_ratings)
    item_ids = np.random.randint(0, n_items, n_ratings)
    ratings = np.random.randint(1, 6, n_ratings)  # Ratings from 1 to 5
    
    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings
    })
    
    # Remove duplicate user-item pairs, keeping the first
    df = df.drop_duplicates(subset=['user_id', 'item_id'])
    
    print(f"Generated {len(df)} ratings for {n_users} users and {n_items} items")
    return df


def train_svd_model(ratings_df, n_factors=50, n_epochs=20):
    """
    Train a simple SVD model using gradient descent.
    This is a simplified version for demonstration purposes.
    """
    print(f"Training SVD model with {n_factors} factors for {n_epochs} epochs...")
    
    n_users = ratings_df['user_id'].max() + 1
    n_items = ratings_df['item_id'].max() + 1
    
    # Initialize user and item factor matrices
    np.random.seed(42)
    user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
    item_factors = np.random.normal(0, 0.1, (n_items, n_factors))
    
    # Simple training loop (simulated)
    learning_rate = 0.01
    regularization = 0.02
    
    for epoch in range(n_epochs):
        # Simulate some computation time
        time.sleep(0.1)
        
        # In a real implementation, you would update factors here
        # For now, we just simulate the computation
        error = 0
        for _, row in ratings_df.iterrows():
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            rating = row['rating']
            
            # Predict rating
            prediction = np.dot(user_factors[user_id], item_factors[item_id])
            error += (rating - prediction) ** 2
            
        rmse = np.sqrt(error / len(ratings_df))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, RMSE: {rmse:.4f}")
    
    print("Model training completed!")
    return user_factors, item_factors


def make_recommendations(user_factors, item_factors, user_id, top_k=10):
    """Generate top-k recommendations for a user."""
    print(f"Generating top-{top_k} recommendations for user {user_id}...")
    
    # Compute predicted ratings for all items
    predictions = np.dot(user_factors[user_id], item_factors.T)
    
    # Get top-k items
    top_items = np.argsort(predictions)[-top_k:][::-1]
    
    print(f"Top {top_k} recommended items: {top_items.tolist()}")
    return top_items


def main():
    """Main function to run the SVD model."""
    print("=" * 50)
    print("SVD Recommender System Model")
    print("=" * 50)
    
    # Generate sample data
    ratings_df = generate_sample_data(n_users=1000, n_items=500)
    
    # Train the model
    user_factors, item_factors = train_svd_model(
        ratings_df,
        n_factors=50,
        n_epochs=20
    )
    
    # Make sample recommendations
    sample_user_id = 0
    recommendations = make_recommendations(
        user_factors,
        item_factors,
        sample_user_id,
        top_k=10
    )
    
    print("=" * 50)
    print("SVD Model execution completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
