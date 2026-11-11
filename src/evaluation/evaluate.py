from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.evaluation.metrics import catalog_coverage


def evaluate_user_recommendations(
    user_id: int | str,
    test_reviews: pd.DataFrame,
    recommended: Iterable,
    k: int = 5,
    relevance_threshold: int = 4,
) -> dict:
    """Avalia uma lista de recomendações para um usuário específico."""
    recommended_list = list(recommended)[:k]
    user_test = test_reviews[test_reviews["AuthorId"] == user_id].copy()
    if user_test.empty:
        return {
            "user_id": user_id,
            "precision@5": 0.0,
            "recall@5": 0.0,
            "ndcg@5": 0.0,
            "precision@10": 0.0,
            "recall@10": 0.0,
            "ndcg@10": 0.0,
            "n_relevant": 0,
        }

    relevant = set(user_test.loc[user_test["Rating"] >= relevance_threshold, "RecipeId"].tolist())
    if not relevant:
        return {
            "user_id": user_id,
            "precision@5": 0.0,
            "recall@5": 0.0,
            "ndcg@5": 0.0,
            "precision@10": 0.0,
            "recall@10": 0.0,
            "ndcg@10": 0.0,
            "n_relevant": 0,
        }

    hits_5 = sum(item in relevant for item in recommended_list[:5])
    hits_10 = sum(item in relevant for item in recommended_list[:10])
    precision_5 = hits_5 / max(len(recommended_list[:5]), 1)
    precision_10 = hits_10 / max(len(recommended_list[:10]), 1)
    recall_5 = hits_5 / len(relevant)
    recall_10 = hits_10 / len(relevant)
    dcg_5 = sum((1 if item in relevant else 0) / (idx + 2) for idx, item in enumerate(recommended_list[:5]))
    ideal_dcg_5 = sum(1 / (idx + 2) for idx in range(min(len(relevant), 5)))
    ndcg_5 = dcg_5 / ideal_dcg_5 if ideal_dcg_5 else 0.0
    dcg_10 = sum((1 if item in relevant else 0) / (idx + 2) for idx, item in enumerate(recommended_list[:10]))
    ideal_dcg_10 = sum(1 / (idx + 2) for idx in range(min(len(relevant), 10)))
    ndcg_10 = dcg_10 / ideal_dcg_10 if ideal_dcg_10 else 0.0
    return {
        "user_id": user_id,
        "precision@5": precision_5,
        "recall@5": recall_5,
        "ndcg@5": ndcg_5,
        "precision@10": precision_10,
        "recall@10": recall_10,
        "ndcg@10": ndcg_10,
        "n_relevant": len(relevant),
    }


def summarize_user_results(results: pd.DataFrame) -> dict:
    """Calcula métricas médias por usuário."""
    if results.empty:
        return {"precision@5": 0.0, "recall@5": 0.0, "ndcg@5": 0.0, "precision@10": 0.0, "recall@10": 0.0, "ndcg@10": 0.0, "catalog_coverage": 0.0}

    precision_5 = results["precision@5"].mean() if "precision@5" in results.columns else 0.0
    recall_5 = results["recall@5"].mean() if "recall@5" in results.columns else 0.0
    ndcg_5 = results["ndcg@5"].mean() if "ndcg@5" in results.columns else 0.0
    precision_10 = results["precision@10"].mean() if "precision@10" in results.columns else 0.0
    recall_10 = results["recall@10"].mean() if "recall@10" in results.columns else 0.0
    ndcg_10 = results["ndcg@10"].mean() if "ndcg@10" in results.columns else 0.0
    recommended_lists = results.get("recommended", pd.Series(dtype=object)).tolist()
    coverage = catalog_coverage(recommended_lists, total_items=max(1, len(set(item for group in recommended_lists for item in (group or []))))) if recommended_lists else 0.0
    return {
        "precision@5": float(precision_5),
        "recall@5": float(recall_5),
        "ndcg@5": float(ndcg_5),
        "precision@10": float(precision_10),
        "recall@10": float(recall_10),
        "ndcg@10": float(ndcg_10),
        "catalog_coverage": float(coverage),
    }


__all__ = ["evaluate_user_recommendations", "summarize_user_results"]
