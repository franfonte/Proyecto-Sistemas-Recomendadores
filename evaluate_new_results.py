"""Augment recommender evaluation results with additional rank-based metrics.

This script loads model predictions and ground-truth interactions for each dataset
size, computes several top-k metrics, and appends them to a copy of the
``results.json`` file, writing the enriched structure to ``results_updated.json``.

UPDATED: Handles String IDs (ISBNs, alphanumeric) instead of enforcing Integers.

Usage:
    python evaluate_new_results.py
    python evaluate_new_results.py --results custom_results.json --output results_augmented.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

TOP_K = 10
FALLBACK_PROBABILITY = 1e-12

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate additional recommendation metrics.")
    parser.add_argument("--results", default="results.json", help="Path to the existing results JSON file.")
    parser.add_argument("--data-dir", default="data", help="Directory containing dataset size folders.")
    parser.add_argument(
        "--output",
        default="results_updated.json",
        help="Destination path for the updated results JSON file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def load_results(path: Path) -> MutableMapping[str, Dict[str, Mapping]]:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_item_popularity(train_df: pd.DataFrame) -> Dict[str, float]:
    """Calculates item popularity based on frequency. Returns Dict[ItemID_Str, Probability]."""
    item_counts = train_df["movieId"].value_counts()
    total_interactions = float(len(train_df))
    if total_interactions == 0:
        return {}
    popularity = (item_counts / total_interactions).to_dict()
    # MODIFICADO: Cast a str para soportar ISBNs
    return {str(item): float(prob) for item, prob in popularity.items()}


def build_user_item_matrix(train_df: pd.DataFrame) -> tuple[csr_matrix, Dict[str, int]]:
    """Builds sparse matrix. Handles string IDs for users and items."""
    if train_df.empty:
        return csr_matrix((0, 0)), {}
    
    # Aseguramos que los IDs sean strings para consistencia
    train_df["userId"] = train_df["userId"].astype(str)
    train_df["movieId"] = train_df["movieId"].astype(str)

    user_ids = pd.Index(train_df["userId"].unique())
    item_ids = pd.Index(train_df["movieId"].unique())

    # MODIFICADO: Las claves ahora son strings
    user_index = {str(user_id): idx for idx, user_id in enumerate(user_ids)}
    item_index = {str(item_id): idx for idx, item_id in enumerate(item_ids)}

    rows = train_df["userId"].map(user_index).to_numpy()
    cols = train_df["movieId"].map(item_index).to_numpy()
    data = np.ones_like(rows, dtype=np.float32)

    matrix = csr_matrix((data, (rows, cols)), shape=(len(user_index), len(item_index)))
    return matrix, item_index


def build_relevance(test_df: pd.DataFrame) -> Dict[str, set[str]]:
    """Builds ground truth dictionary. Keys and Values are Strings."""
    if test_df.empty:
        return {}
    
    # Asegurar consistencia de tipos
    test_df["userId"] = test_df["userId"].astype(str)
    test_df["movieId"] = test_df["movieId"].astype(str)

    positive = test_df[test_df["rating"] > 0]
    # MODIFICADO: Cast a str dentro del set comprehension
    grouped = positive.groupby("userId")["movieId"].apply(lambda items: set(str(x) for x in items))
    return {str(user): items for user, items in grouped.items()}


def deduplicate_preserve_order(items: Sequence[Any]) -> List[Any]:
    seen = set()
    deduped = []
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def build_recommendations(predictions_df: pd.DataFrame, top_k: int) -> Dict[str, List[str]]:
    """Extracts top-k recommendations. Returns Dict[UserID_Str, List[ItemID_Str]]."""
    if predictions_df.empty:
        return {}

    required_columns = {"userId", "movieId", "prediction"}
    missing_columns = required_columns - set(predictions_df.columns)
    if missing_columns:
        raise ValueError(f"Predictions file is missing required columns: {sorted(missing_columns)}")
    
    # Asegurar que IDs sean strings antes de procesar
    predictions_df["userId"] = predictions_df["userId"].astype(str)
    predictions_df["movieId"] = predictions_df["movieId"].astype(str)

    sorted_predictions = predictions_df.sort_values(["userId", "prediction"], ascending=[True, False])
    sorted_predictions = sorted_predictions.drop_duplicates(subset=["userId", "movieId"], keep="first")

    top_predictions = sorted_predictions.groupby("userId").head(top_k)
    recommendation_lists = top_predictions.groupby("userId")["movieId"].apply(list)
    recommendations: Dict[str, List[str]] = {}

    for user_id, recs in recommendation_lists.items():
        # MODIFICADO: Cast a str explícito
        cleaned = deduplicate_preserve_order([str(item) for item in recs])[:top_k]
        if cleaned:
            recommendations[str(user_id)] = cleaned
    return recommendations


def compute_ils(
    item_list: Sequence[str],
    similarity_matrix: sparse.spmatrix | np.ndarray,
    item_index_map: Mapping[str, int],
) -> float:
    indices = [item_index_map.get(item) for item in item_list if item in item_index_map]
    if len(indices) < 2:
        return 0.0

    if sparse.issparse(similarity_matrix):
        # Slicing eficiente para matrices dispersas
        submatrix = similarity_matrix[indices][:, indices].toarray()
    else:
        submatrix = similarity_matrix[np.ix_(indices, indices)]

    np.fill_diagonal(submatrix, 0.0)
    pair_count = len(indices) * (len(indices) - 1) / 2
    if pair_count == 0:
        return 0.0
    
    avg_similarity = float(np.triu(submatrix, k=1).sum() / pair_count)
    return max(0.0, 1.0 - avg_similarity)


def evaluate_recommendations(
    recommendations: Mapping[str, Sequence[str]],
    relevant_items: Mapping[str, set[str]],
    popularity: Mapping[str, float],
    item_index_map: Mapping[str, int],
    similarity_matrix: sparse.spmatrix | np.ndarray,
    top_k: int,
) -> Dict[str, float] | None:
    hit_rates: List[float] = []
    reciprocal_ranks: List[float] = []
    average_precisions: List[float] = []
    novelty_scores: List[float] = []
    diversity_scores: List[float] = []

    for user_id, recs in recommendations.items():
        relevant = relevant_items.get(user_id)
        if not relevant:
            continue

        rec_list = recs[:top_k]
        if not rec_list:
            continue

        # Hit Rate
        hit = any(item in relevant for item in rec_list)
        hit_rates.append(1.0 if hit else 0.0)

        # MRR
        reciprocal = 0.0
        for rank, item in enumerate(rec_list, start=1):
            if item in relevant:
                reciprocal = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal)

        # MAP
        hits = 0
        precision_sum = 0.0
        for rank, item in enumerate(rec_list, start=1):
            if item in relevant:
                hits += 1
                precision_sum += hits / rank
        denominator = min(len(relevant), top_k)
        average_precisions.append(precision_sum / denominator if denominator else 0.0)

        # Novelty
        novelty_values = []
        for item in rec_list:
            prob = popularity.get(item, 0.0)
            probability = prob if prob > 0.0 else FALLBACK_PROBABILITY
            novelty_values.append(-math.log2(probability))
        if novelty_values:
            novelty_scores.append(float(sum(novelty_values) / len(novelty_values)))

        # Diversity
        diversity_scores.append(compute_ils(rec_list, similarity_matrix, item_index_map))

    evaluated_users = len(hit_rates)
    if evaluated_users == 0:
        return None

    metrics = {
        "hit_rate_at_10": float(sum(hit_rates) / evaluated_users),
        "mrr_at_10": float(sum(reciprocal_ranks) / evaluated_users),
        "map_at_10": float(sum(average_precisions) / evaluated_users),
        "novelty_at_10": float(sum(novelty_scores) / len(novelty_scores)) if novelty_scores else 0.0,
        "diversity_at_10": float(sum(diversity_scores) / len(diversity_scores)) if diversity_scores else 0.0,
    }
    return metrics


def prepare_dataset_artifacts(
    dataset_size: str,
    data_dir: Path,
) -> tuple[Dict[str, set[str]], Dict[str, float], Dict[str, int], sparse.spmatrix | np.ndarray] | None:
    dataset_path = data_dir / dataset_size
    train_path = dataset_path / "train.csv"
    test_path = dataset_path / "test.csv"

    if not train_path.exists() or not test_path.exists():
        missing = []
        if not train_path.exists():
            missing.append(str(train_path))
        if not test_path.exists():
            missing.append(str(test_path))
        LOGGER.warning("Skipping dataset size %s due to missing files: %s", dataset_size, ", ".join(missing))
        return None

    try:
        # Cargamos CSVs. Forzaremos el tipo str más adelante para mayor seguridad.
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except Exception as exc:
        LOGGER.warning("Skipping dataset size %s due to read error: %s", dataset_size, exc)
        return None

    # Compute Artifacts
    popularity = compute_item_popularity(train_df)
    user_item_matrix, item_index_map = build_user_item_matrix(train_df)
    relevance = build_relevance(test_df)

    if user_item_matrix.shape[1] == 0 or not item_index_map:
        LOGGER.warning("Skipping dataset size %s because the training data has no items.", dataset_size)
        return None

    LOGGER.debug(
        "Precomputing item-item similarity for dataset %s (%d users x %d items)",
        dataset_size,
        user_item_matrix.shape[0],
        user_item_matrix.shape[1],
    )

    # Calculate Cosine Similarity on sparse matrix
    similarity_matrix = cosine_similarity(user_item_matrix.T, dense_output=False)
    return relevance, popularity, item_index_map, similarity_matrix


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    results_path = Path(args.results)
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    try:
        original_results = load_results(results_path)
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        return

    updated_results = copy.deepcopy(original_results)

    for dataset_size, models in original_results.items():
        LOGGER.info("Processing dataset size %s", dataset_size)
        artifacts = prepare_dataset_artifacts(str(dataset_size), data_dir)
        if artifacts is None:
            LOGGER.info("No evaluation artifacts created for dataset size %s, skipping.", dataset_size)
            continue

        relevance, popularity, item_index_map, similarity_matrix = artifacts

        for model_name in models.keys():
            predictions_path = data_dir / str(dataset_size) / f"{model_name}_predictions.csv"
            if not predictions_path.exists():
                LOGGER.warning(
                    "Missing predictions for model %s at dataset size %s: %s",
                    model_name,
                    dataset_size,
                    predictions_path,
                )
                continue

            try:
                predictions_df = pd.read_csv(predictions_path)
            except Exception as exc:
                LOGGER.warning(
                    "Failed to read predictions for model %s at dataset size %s: %s",
                    model_name,
                    dataset_size,
                    exc,
                )
                continue

            try:
                recommendations = build_recommendations(predictions_df, TOP_K)
            except ValueError as exc:
                LOGGER.warning(
                    "Invalid predictions for model %s at dataset size %s: %s",
                    model_name,
                    dataset_size,
                    exc,
                )
                continue

            metrics = evaluate_recommendations(
                recommendations,
                relevance,
                popularity,
                item_index_map,
                similarity_matrix,
                TOP_K,
            )

            if metrics is None:
                LOGGER.info(
                    "Insufficient data to compute metrics for model %s at dataset size %s, skipping.",
                    model_name,
                    dataset_size,
                )
                continue

            LOGGER.debug("Metrics for %s on dataset %s: %s", model_name, dataset_size, metrics)
            updated_results[str(dataset_size)][model_name].setdefault("performance_metrics", {})
            updated_results[str(dataset_size)][model_name]["performance_metrics"].update(metrics)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(updated_results, handle, indent=4)

    LOGGER.info("Saved updated evaluation results to %s", output_path)


if __name__ == "__main__":
    main()