from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_user_profile(
    reviews_df: pd.DataFrame,
    recipes_df: pd.DataFrame,
    tfidf_builder: Any,
    rating_weight: float = 2.0,
    positive_threshold: int = 4,
    neutral_threshold: int = 2,
    min_history: int = 1,
) -> tuple[np.ndarray | None, pd.DataFrame]:
    """Cria um perfil textual para um usuário a partir do histórico de avaliações.

    A nota 0, presente no arquivo real apesar da escala usual de 1 a 5, é tratada
    como a menor avaliação (-2). Notas 1-5 recebem pesos -2, -1, 0, +1 e +2.
    Avaliações repetidas do mesmo usuário para a mesma receita são agregadas pela
    média antes da ponderação, evitando contar a mesma receita várias vezes.
    """
    if reviews_df.empty or recipes_df.empty:
        return None, pd.DataFrame()

    merged = reviews_df.merge(recipes_df[["RecipeId", "combined_text"]], on="RecipeId", how="inner")
    if merged.empty:
        return None, pd.DataFrame()

    if "AuthorId" in merged.columns:
        merged = merged.groupby(["AuthorId", "RecipeId"], as_index=False, dropna=False).agg(
            Rating=("Rating", "mean"),
            combined_text=("combined_text", "first"),
        )
    else:
        merged = merged.groupby("RecipeId", as_index=False, dropna=False).agg(
            Rating=("Rating", "mean"),
            combined_text=("combined_text", "first"),
        )
    if len(merged) < min_history:
        return None, merged

    def weight_for_rating(rating: float) -> float:
        if pd.isna(rating) or rating == 3:
            return 0.0
        if rating >= positive_threshold:
            return rating_weight if rating >= 5 else 1.0
        if rating <= 0:
            return -2.0
        if rating <= neutral_threshold:
            return -2.0 if rating <= 1 else -1.0
        return 0.0

    merged["score_weight"] = merged["Rating"].apply(weight_for_rating)
    weighted_history = merged[merged["score_weight"] != 0].copy()
    if weighted_history.empty or len(weighted_history) < min_history:
        return None, merged

    weighted_vectors = [
        tfidf_builder.get_vector_for_recipe(row.RecipeId) * float(row.score_weight)
        for row in weighted_history.itertuples(index=False)
    ]
    user_vector = np.sum(weighted_vectors, axis=0)
    norm = float(np.linalg.norm(user_vector))
    if norm <= 1e-12:
        return None, merged
    user_vector /= norm
    return user_vector, merged


def summarize_user_history(reviews_df: pd.DataFrame) -> dict[str, Any]:
    """Retorna resumo do histórico para a interface e para a análise qualitativa."""
    if reviews_df.empty:
        return {
            "count": 0,
            "average_rating": 0.0,
            "positive": 0,
            "negative": 0,
        }
    avg = float(reviews_df["Rating"].mean())
    return {
        "count": int(len(reviews_df)),
        "average_rating": avg,
        "positive": int((reviews_df["Rating"] >= 4).sum()),
        "negative": int((reviews_df["Rating"] <= 2).sum()),
    }


__all__ = ["build_user_profile", "summarize_user_history"]
