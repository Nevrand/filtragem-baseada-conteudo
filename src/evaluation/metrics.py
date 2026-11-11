from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def precision_at_k(relevant_items: set | list, recommended_items: Iterable, k: int = 5) -> float:
    """Precision@k: proporção de itens recomendados e relevantes na lista."""
    recommended = list(recommended_items)[:k]
    if not recommended:
        return 0.0
    relevant = set(relevant_items)
    hits = sum(1 for item in recommended if item in relevant)
    return hits / len(recommended)


def recall_at_k(relevant_items: set | list, recommended_items: Iterable, k: int = 5) -> float:
    """Recall@k: itens relevantes capturados pela recomendação."""
    relevant = set(relevant_items)
    if not relevant:
        return 0.0
    recommended = list(recommended_items)[:k]
    hits = sum(1 for item in recommended if item in relevant)
    return hits / len(relevant)


def dcg_at_k(scores: Iterable[float], k: int = 5) -> float:
    """Discounted Cumulative Gain para uma lista ordenada de pontuações."""
    score_list = list(scores)[:k]
    if not score_list:
        return 0.0
    return sum((2 ** float(score) - 1) / np.log2(idx + 2) for idx, score in enumerate(score_list))


def ndcg_at_k(relevant_scores: Iterable[float], recommended_scores: Iterable[float], k: int = 5) -> float:
    """nDCG@k para uma lista de recomendações ordenadas pelo score."""
    rel = list(relevant_scores)[:k]
    rec = list(recommended_scores)[:k]
    if not rel or not rec:
        return 0.0
    ideal = dcg_at_k(sorted(rel, reverse=True), k)
    return 0.0 if ideal == 0 else dcg_at_k(rec, k) / ideal


def catalog_coverage(recommended_lists: Iterable[Iterable], total_items: int) -> float:
    """Cobertura do catálogo: porcentagem de itens únicos recomendados."""
    seen = set()
    for items in recommended_lists:
        seen.update(items)
    if total_items == 0:
        return 0.0
    return len(seen) / total_items


def average_metrics_by_user(results: pd.DataFrame) -> pd.DataFrame:
    """Calcula médias por usuário a partir de uma tabela de resultados."""
    if results.empty:
        return pd.DataFrame(columns=["user_id", "precision@5", "recall@5", "ndcg@5", "precision@10", "recall@10", "ndcg@10"])
    grouped = results.groupby("user_id", as_index=False).mean(numeric_only=True)
    return grouped


__all__ = [
    "precision_at_k",
    "recall_at_k",
    "dcg_at_k",
    "ndcg_at_k",
    "catalog_coverage",
    "average_metrics_by_user",
]
