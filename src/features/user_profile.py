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
    """Cria um perfil usando avaliações por estrelas, sem descartar interações.

    Rating 0 indica uma interação sem avaliação por estrelas, não uma nota.
    Somente notas válidas de 1 a 5 são agregadas e ponderadas; o histórico
    retornado também mantém receitas sem nota para preservar os itens visitados.
    """
    if reviews_df.empty or recipes_df.empty:
        return None, pd.DataFrame()

    merged = reviews_df.merge(recipes_df[["RecipeId", "combined_text"]], on="RecipeId", how="inner")
    if merged.empty:
        return None, pd.DataFrame()

    merged["_numeric_rating"] = pd.to_numeric(merged["Rating"], errors="coerce")
    merged["_valid_rating"] = merged["_numeric_rating"].between(1, 5)
    merged["_unrated_interaction"] = merged["_numeric_rating"].eq(0)
    if "AuthorId" in merged.columns:
        group_columns = ["AuthorId", "RecipeId"]
    else:
        group_columns = ["RecipeId"]
    history = merged.groupby(group_columns, as_index=False, dropna=False).agg(
        Rating=("_numeric_rating", lambda values: values[values.between(1, 5)].mean()),
        interaction_count=("RecipeId", "size"),
        valid_rating_count=("_valid_rating", "sum"),
        unrated_interaction_count=("_unrated_interaction", "sum"),
        combined_text=("combined_text", "first"),
    )
    history.loc[
        history["Rating"].isna() & history["unrated_interaction_count"].gt(0),
        "Rating",
    ] = 0.0

    def weight_for_rating(rating: float) -> float:
        if pd.isna(rating) or not 1 <= rating <= 5 or rating == 3:
            return 0.0
        if rating >= positive_threshold:
            return rating_weight if rating >= 5 else 1.0
        if rating <= neutral_threshold:
            return -2.0 if rating <= 1 else -1.0
        return 0.0

    history["score_weight"] = history["Rating"].apply(weight_for_rating)
    if len(history) < min_history:
        return None, history

    weighted_history = history[history["score_weight"] != 0].copy()
    if weighted_history.empty or len(weighted_history) < min_history:
        return None, history

    weighted_vectors = [
        tfidf_builder.get_vector_for_recipe(row.RecipeId) * float(row.score_weight)
        for row in weighted_history.itertuples(index=False)
    ]
    user_vector = np.sum(weighted_vectors, axis=0)
    norm = float(np.linalg.norm(user_vector))
    if norm <= 1e-12:
        return None, history
    user_vector /= norm
    return user_vector, history


def summarize_user_history(reviews_df: pd.DataFrame) -> dict[str, Any]:
    """Retorna resumo do histórico para a interface e para a análise qualitativa."""
    interaction_count = int(len(reviews_df))
    ratings = (
        pd.to_numeric(reviews_df["Rating"], errors="coerce")
        if "Rating" in reviews_df.columns
        else pd.Series(dtype=float)
    )
    valid_ratings = ratings[ratings.between(1, 5)]
    if reviews_df.empty:
        return {
            "count": 0,
            "interaction_count": 0,
            "valid_rating_count": 0,
            "average_rating": None,
            "positive": 0,
            "negative": 0,
        }
    return {
        "count": interaction_count,
        "interaction_count": interaction_count,
        "valid_rating_count": int(len(valid_ratings)),
        "average_rating": float(valid_ratings.mean()) if not valid_ratings.empty else None,
        "positive": int(valid_ratings.between(4, 5).sum()),
        "negative": int(valid_ratings.between(1, 2).sum()),
    }


__all__ = ["build_user_profile", "summarize_user_history"]
