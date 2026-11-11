from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def precision_at_k(relevant_items: set | list, recommended_items: Iterable, k: int = 5) -> float:
    """Precision@K binária; o denominador é K, mesmo se a lista for curta."""
    if k <= 0:
        raise ValueError("k deve ser maior que zero.")
    relevant = set(relevant_items)
    recommended = list(recommended_items)[:k]
    return sum(item in relevant for item in recommended) / k


def recall_at_k(relevant_items: set | list, recommended_items: Iterable, k: int = 5) -> float:
    """Recall@K binária; conjunto relevante vazio retorna zero."""
    if k <= 0:
        raise ValueError("k deve ser maior que zero.")
    relevant = set(relevant_items)
    if not relevant:
        return 0.0
    recommended = list(recommended_items)[:k]
    return sum(item in relevant for item in recommended) / len(relevant)


def dcg_at_k(relevance_scores: Iterable[float], k: int = 5) -> float:
    """DCG com ganho binário/linear: sum(rel_i / log2(i + 2))."""
    if k <= 0:
        raise ValueError("k deve ser maior que zero.")
    scores = list(relevance_scores)[:k]
    return float(sum(float(relevance) / np.log2(index + 2) for index, relevance in enumerate(scores)))


def ndcg_at_k(relevant_items: set | list, recommended_items: Iterable, k: int = 5) -> float:
    """nDCG binária, limitada a [0, 1], com IDCG calculado pelo total relevante."""
    if k <= 0:
        raise ValueError("k deve ser maior que zero.")
    relevant = set(relevant_items)
    if not relevant:
        return 0.0
    recommended = list(recommended_items)[:k]
    dcg = dcg_at_k([int(item in relevant) for item in recommended], k)
    idcg = dcg_at_k([1] * min(len(relevant), k), k)
    return min(1.0, dcg / idcg) if idcg else 0.0


def catalog_coverage(recommended_lists: Iterable[Iterable], total_items: int) -> float:
    """Cobertura: únicos recomendados sobre o total de itens elegíveis."""
    if total_items <= 0:
        return 0.0
    seen = {item for items in recommended_lists for item in items}
    return len(seen) / total_items


def average_metrics_by_user(results: pd.DataFrame) -> pd.DataFrame:
    """Calcula médias numéricas por usuário."""
    if results.empty:
        return pd.DataFrame(columns=["user_id"])
    return results.groupby("user_id", as_index=False).mean(numeric_only=True)


__all__ = [
    "precision_at_k",
    "recall_at_k",
    "dcg_at_k",
    "ndcg_at_k",
    "catalog_coverage",
    "average_metrics_by_user",
]
