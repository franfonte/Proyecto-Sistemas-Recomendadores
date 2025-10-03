import pandas as pd
import numpy as np

def calculate_rmse(merged_df):
    """
    Calculates the Root Mean Squared Error on the inner join of predictions and ground truth.
    This metric remains correct as it only evaluates predicted ratings for items in the test set.
    """
    if 'rating' not in merged_df.columns or 'prediction' not in merged_df.columns:
        return np.nan
    return np.sqrt(((merged_df['rating'] - merged_df['prediction'])**2).mean())

def calculate_ranking_metrics(predictions_df, test_df, k=10, relevance_threshold=4.0):
    """
    Calculates Precision@k, Recall@k, and nDCG@k.
    This function now correctly handles the full list of predictions.
    """
    # Group predictions and test data by user
    user_predictions = predictions_df.groupby('userId')
    user_true_relevants = test_df[test_df['rating'] >= relevance_threshold].groupby('userId')['movieId'].apply(list)

    precisions = []
    recalls = []
    ndcgs = []

    for user_id, group in user_predictions:
        # Skip users that are not in the test set ground truth
        if user_id not in user_true_relevants:
            continue

        # Sort user's predictions and get top k
        top_k_preds = group.sort_values(by='prediction', ascending=False).head(k)
        recommended_items = top_k_preds['movieId'].tolist()
        
        # Get true relevant items for this user
        true_items = user_true_relevants[user_id]
        
        # --- Calculate Metrics ---
        hits = len(set(recommended_items) & set(true_items))
        
        # Precision@k
        precision = hits / k if k > 0 else 0
        precisions.append(precision)
        
        # Recall@k
        recall = hits / len(true_items) if len(true_items) > 0 else 0
        recalls.append(recall)

        # nDCG@k
        relevance_map = {movie_id: 1 if movie_id in true_items else 0 for movie_id in recommended_items}
        dcg = sum(relevance_map.get(rec, 0) / np.log2(i + 2) for i, rec in enumerate(recommended_items))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)

    # Return the average of metrics across all users
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_ndcg = np.mean(ndcgs) if ndcgs else 0
    
    return avg_precision, avg_recall, avg_ndcg

