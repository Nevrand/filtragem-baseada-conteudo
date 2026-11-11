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

    A estratégia adota pesos angulares: avaliações positivas elevam a importância dos
    termos, avaliações negativas reduzem, e avaliações neutras têm efeito menor.
    """
    if reviews_df.empty or recipes_df.empty:
        return None, pd.DataFrame()

    merged = reviews_df.merge(recipes_df[["RecipeId", "combined_text"]], on="RecipeId", how="inner")
    if merged.empty:
        return None, pd.DataFrame()

    merged["score_weight"] = merged["Rating"].apply(
        lambda rating: (
            rating_weight if rating >= positive_threshold else
            0.25 if rating <= neutral_threshold else
            0.0
        )
    )
    merged = merged[merged["score_weight"] > 0].copy()
    if merged.empty:
        return None, pd.DataFrame()

    user_vector = None
    for _, row in merged.iterrows():
        recipe_vector = tfidf_builder.get_vector_for_recipe(row["RecipeId"])
        weight = float(row["score_weight"])
        weighted_vector = recipe_vector * weight
        if user_vector is None:
            user_vector = weighted_vector.copy()
        else:
            user_vector += weighted_vector

    if user_vector is None:
        return None, merged

    user_vector = user_vector / (np.linalg.norm(user_vector) + 1e-12)
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
