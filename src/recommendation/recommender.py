from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.data.preprocess import coerce_ingredient_list
from src.features.user_profile import build_user_profile
from src.recommendation.filters import filter_candidates


@dataclass
class RecipeRecommender:
    """Executa o ranking de receitas com base em conteúdo e filtros."""

    recipes_df: pd.DataFrame
    reviews_df: pd.DataFrame
    tfidf_builder: Any
    user_id: int | str | None = None
    candidate_limit: int = 1000
    top_n: int = 5
    diversity_lambda: float = 0.0
    use_mmr: bool = False

    def _ensure_user_ratings(self) -> pd.DataFrame:
        if self.reviews_df.empty:
            return pd.DataFrame(columns=["RecipeId", "AuthorId", "Rating"])
        return self.reviews_df[self.reviews_df["AuthorId"] == self.user_id].copy() if self.user_id is not None else self.reviews_df.head(0).copy()

    def _prepare_candidates(self, max_minutes: float | int | None = None, avoided_ingredients: Sequence[str] | None = None, available_ingredients: Sequence[str] | None = None) -> pd.DataFrame:
        candidates = self.recipes_df.copy()
        user_ratings = self._ensure_user_ratings()
        return filter_candidates(
            candidates,
            user_ratings,
            max_minutes=max_minutes,
            avoided_ingredients=avoided_ingredients,
            available_ingredients=available_ingredients,
            ingredient_column="RecipeIngredientParts",
            recipe_id_column="RecipeId",
        )

    def compute_scores(self, user_vector: np.ndarray | None, candidate_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Calcula scores e retorna lista de receitas candidatas ordenadas."""
        if user_vector is None:
            return pd.DataFrame(columns=["RecipeId", "score", "explanation"])
        if candidate_df is None:
            candidate_df = self._prepare_candidates()

        if candidate_df.empty:
            return pd.DataFrame(columns=["RecipeId", "score", "explanation"])

        ids = candidate_df["RecipeId"].astype(str).tolist()
        matrix = self.tfidf_builder.matrix[[self.tfidf_builder.index_by_id[str(recipe_id)] for recipe_id in ids if str(recipe_id) in self.tfidf_builder.index_by_id], :]
        scores = (matrix @ user_vector.reshape(-1, 1)).ravel()
        ranking = candidate_df.copy()
        ranking["score"] = scores
        ranking = ranking.sort_values("score", ascending=False)
        ranking["explanation"] = ranking.apply(self._build_explanation, axis=1)
        return ranking.reset_index(drop=True)

    def _build_explanation(self, row: pd.Series) -> str:
        """Cria uma justificativa textual para a recomendação."""
        ingredients = coerce_ingredient_list(row.get("RecipeIngredientParts", []))
        top_ingredients = " , ".join(ingredients[:4]) if ingredients else "ingredientes e termos relacionados"
        return (
            f"A receita combina com ingredientes e termos frequentemente presentes em suas avaliações positivas, "
            f"especialmente {top_ingredients}."
        )

    def recommend_top_n(
        self,
        user_id: int | str | None = None,
        max_minutes: float | int | None = None,
        avoided_ingredients: Sequence[str] | None = None,
        available_ingredients: Sequence[str] | None = None,
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """Retorna as melhores receitas para o usuário seguindo o ranking e filtros."""
        self.user_id = user_id if user_id is not None else self.user_id
        candidates = self._prepare_candidates(max_minutes=max_minutes, avoided_ingredients=avoided_ingredients, available_ingredients=available_ingredients)
        if candidates.empty:
            return pd.DataFrame(columns=["RecipeId", "score", "explanation"])

        user_vector, _ = build_user_profile(self.reviews_df[self.reviews_df["AuthorId"] == self.user_id].copy() if self.user_id is not None else self.reviews_df.head(0).copy(), self.recipes_df, self.tfidf_builder)
        if user_vector is None:
            return pd.DataFrame(columns=["RecipeId", "score", "explanation"])

        ranked = self.compute_scores(user_vector, candidate_df=candidates)
        output = ranked.head(top_n if top_n is not None else self.top_n).copy()
        output["rank"] = range(1, len(output) + 1)
        return output.reset_index(drop=True)

    def mmr_diversify(self, ranked_df: pd.DataFrame, lambda_param: float | None = None) -> pd.DataFrame:
        """Reordena as recomendações usando MMR para equilibrar relevância e diversidade."""
        if ranked_df.empty or lambda_param is None:
            return ranked_df.copy()

        selected = []
        remaining = ranked_df.copy().reset_index(drop=True)
        if remaining.empty:
            return remaining

        for _ in range(min(len(remaining), 5)):
            best_idx = None
            best_value = -np.inf
            first_choice = remaining.iloc[0]
            for idx, row in remaining.iterrows():
                novelty = 0.0
                if selected:
                    similarity_scores = []
                    for _, selected_row in selected:
                        if "_vector" in row and "_vector" in selected_row and row["_vector"] is not None and selected_row["_vector"] is not None:
                            similarity_scores.append(float(np.dot(np.asarray(row["_vector"]), np.asarray(selected_row["_vector"]))))
                    novelty = max(similarity_scores) if similarity_scores else 0.0
                value = lambda_param * float(row.get("score", 0.0)) - (1 - lambda_param) * novelty
                if value > best_value:
                    best_value = value
                    best_idx = idx
            if best_idx is None:
                break
            selected.append((best_idx, remaining.loc[best_idx].copy()))
            remaining = remaining.drop(index=best_idx)
        result = pd.DataFrame([entry[1] for entry in selected])
        return result.reset_index(drop=True)


__all__ = ["RecipeRecommender"]
