from __future__ import annotations

import numpy as np
import pandas as pd


def random_recommendation(candidates: pd.DataFrame, user_id: int | str | None = None, top_n: int = 5, random_state: int = 42) -> list:
    """Gera recomendação aleatória usando semente fixa para reprodutibilidade."""
    if candidates.empty:
        return []
    rng = np.random.default_rng(random_state)
    sample = candidates.sample(n=min(len(candidates), top_n), random_state=rng.integers(0, 10_000))
    return sample["RecipeId"].astype(int).tolist()


def popularity_recommendation(reviews_df: pd.DataFrame, recipes_df: pd.DataFrame, top_n: int = 5) -> list:
    """Recomenda itens mais populares no conjunto de treino."""
    if reviews_df.empty:
        return []
    counts = reviews_df.groupby("RecipeId").size().reset_index(name="count")
    scores = counts.merge(recipes_df[["RecipeId"]], on="RecipeId", how="inner")
    if scores.empty:
        return []
    return scores.sort_values("count", ascending=False)["RecipeId"].head(top_n).astype(int).tolist()


__all__ = ["random_recommendation", "popularity_recommendation"]
