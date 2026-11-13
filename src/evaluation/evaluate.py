from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.evaluation.metrics import catalog_coverage, ndcg_at_k, precision_at_k, recall_at_k

METRIC_COLUMNS = (
    "precision@5",
    "recall@5",
    "ndcg@5",
    "precision@10",
    "recall@10",
    "ndcg@10",
)


def evaluate_user_recommendations(
    user_id: int | str,
    test_reviews: pd.DataFrame,
    recommended: Iterable,
    k: int = 10,
    relevance_threshold: int = 4,
    user_column: str = "AuthorId",
    recipe_column: str = "RecipeId",
    rating_column: str = "Rating",
    relevant_items: Iterable | None = None,
) -> dict:
    """Calcula métricas @5/@10 usando o conjunto relevante elegível fornecido."""
    user_test = test_reviews[test_reviews[user_column] == user_id]
    if relevant_items is None:
        ratings = pd.to_numeric(user_test[rating_column], errors="coerce")
        relevant = set(
            user_test.loc[
                ratings.between(relevance_threshold, 5),
                recipe_column,
            ].dropna().tolist()
        )
    else:
        relevant = set(relevant_items)
    recommendations = list(recommended)[:max(k, 10)]
    has_test = not user_test.empty
    has_relevant = bool(relevant)
    reason = None if has_relevant else ("no_test_reviews" if not has_test else "no_relevant_items")
    result = {
        "user_id": user_id,
        "evaluated": has_relevant,
        "exclusion_reason": reason,
        "n_test_reviews": int(len(user_test)),
        "n_relevant": len(relevant),
        "n_recommendations": len(recommendations),
        "recommended": recommendations,
    }
    for metric in METRIC_COLUMNS:
        result[metric] = 0.0
    if has_relevant:
        for cutoff in (5, 10):
            items = recommendations[:cutoff]
            result[f"precision@{cutoff}"] = precision_at_k(relevant, items, cutoff)
            result[f"recall@{cutoff}"] = recall_at_k(relevant, items, cutoff)
            result[f"ndcg@{cutoff}"] = ndcg_at_k(relevant, items, cutoff)
    return result


def summarize_user_results(
    results: pd.DataFrame,
    total_eligible_items: int,
    total_users_considered: int | None = None,
) -> dict:
    """Resume métricas somente entre usuários com ao menos um positivo no teste.

    Usuários sem nota >= 4 no teste são contados como excluídos e não entram nas
    médias. A quantidade e o motivo continuam preservados em ``metrics_by_user.csv``.
    """
    if results.empty:
        results = pd.DataFrame(columns=["evaluated", "n_test_reviews", "n_recommendations", "recommended", *METRIC_COLUMNS])
    evaluable = results[results.get("evaluated", pd.Series(dtype=bool)).fillna(False)]
    excluded = results[~results.get("evaluated", pd.Series(dtype=bool)).fillna(False)]
    mean_metrics = {
        metric: float(evaluable[metric].mean()) if not evaluable.empty and metric in evaluable else 0.0
        for metric in METRIC_COLUMNS
    }
    recommended_lists = evaluable["recommended"].tolist() if "recommended" in evaluable.columns else []
    summary = {
        **mean_metrics,
        "coverage": catalog_coverage(recommended_lists, total_eligible_items),
        "total_eligible_items": int(total_eligible_items),
        "evaluated_users": int(len(evaluable)),
        "excluded_users": int(len(excluded)),
        "users_considered": int(total_users_considered if total_users_considered is not None else len(results)),
        "mean_test_reviews_per_user": float(evaluable["n_test_reviews"].mean()) if not evaluable.empty else 0.0,
        "mean_recommendations_per_user": float(evaluable["n_recommendations"].mean()) if not evaluable.empty else 0.0,
    }
    if "exclusion_reason" in excluded.columns:
        summary["excluded_reasons"] = excluded["exclusion_reason"].value_counts().to_dict()
    else:
        summary["excluded_reasons"] = {}
    return summary


__all__ = ["evaluate_user_recommendations", "summarize_user_results", "METRIC_COLUMNS"]
