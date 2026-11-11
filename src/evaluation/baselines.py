from __future__ import annotations

import numpy as np
import pandas as pd


def _exclude_seen(candidates: pd.DataFrame, train_reviews: pd.DataFrame | None) -> pd.DataFrame:
    if train_reviews is None or train_reviews.empty:
        return candidates
    seen = set(train_reviews["RecipeId"].dropna())
    return candidates[~candidates["RecipeId"].isin(seen)]


def random_recommendation(
    candidates: pd.DataFrame,
    user_id: int | str | None = None,
    top_n: int = 5,
    random_state: int = 42,
    train_reviews: pd.DataFrame | None = None,
) -> list:
    """Gera uma lista aleatória determinística a partir das candidatas elegíveis."""
    if candidates.empty:
        return []
    eligible = _exclude_seen(candidates, train_reviews)
    if eligible.empty:
        return []
    user_offset = int(user_id) if isinstance(user_id, (int, np.integer)) else sum(map(ord, str(user_id or "")))
    rng = np.random.default_rng(random_state + user_offset)
    indices = rng.permutation(len(eligible))[:top_n]
    return eligible.iloc[indices]["RecipeId"].tolist()


def popularity_recommendation(
    reviews_df: pd.DataFrame,
    recipes_df: pd.DataFrame,
    top_n: int = 5,
    train_reviews: pd.DataFrame | None = None,
) -> list:
    """Recomenda as candidatas com mais avaliações no conjunto recebido (treino)."""
    if reviews_df.empty:
        return []
    eligible = _exclude_seen(recipes_df, train_reviews)
    if eligible.empty:
        return []
    counts = reviews_df.groupby("RecipeId", dropna=True).size().rename("popularity")
    scores = eligible[["RecipeId"]].copy()
    scores["popularity"] = scores["RecipeId"].map(counts).fillna(0)
    return scores.sort_values(["popularity", "RecipeId"], ascending=[False, True])["RecipeId"].head(top_n).tolist()


__all__ = ["random_recommendation", "popularity_recommendation"]
