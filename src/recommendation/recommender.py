from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data.preprocess import coerce_ingredient_list
from src.features.text_features import recipe_id_key
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
    _active_profile: np.ndarray | None = None

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

    def compute_scores(
        self,
        user_vector: np.ndarray | None,
        candidate_df: pd.DataFrame | None = None,
        top_n: int | None = None,
        explain: bool = True,
    ) -> pd.DataFrame:
        """Calcula scores e retorna lista de receitas candidatas ordenadas."""
        if user_vector is None:
            return pd.DataFrame(columns=["RecipeId", "score", "explanation"])
        self._active_profile = np.asarray(user_vector, dtype=float).ravel()
        if candidate_df is None:
            candidate_df = self._prepare_candidates()

        if candidate_df.empty:
            return pd.DataFrame(columns=["RecipeId", "score", "explanation"])
        if top_n is not None and top_n < 0:
            raise ValueError("top_n não pode ser negativo.")
        candidate_df = candidate_df.drop_duplicates(subset=["RecipeId"], keep="first")

        positions = [
            self.tfidf_builder.index_by_id.get(recipe_id_key(recipe_id))
            for recipe_id in candidate_df["RecipeId"]
        ]
        known = [position is not None for position in positions]
        ranking = candidate_df.loc[known].copy()
        matrix = self.tfidf_builder.matrix[[position for position in positions if position is not None], :]
        if ranking.empty:
            return pd.DataFrame(columns=["RecipeId", "score", "explanation"])
        scores = np.asarray(matrix @ self._active_profile).ravel()
        if top_n is not None and 0 <= top_n < len(scores):
            selected = np.argpartition(scores, -top_n)[-top_n:] if top_n else np.array([], dtype=int)
            ranking = ranking.iloc[selected].copy()
            scores = scores[selected]
        ranking["score"] = scores
        ranking = ranking.sort_values(["score", "RecipeId"], ascending=[False, True], kind="mergesort")
        if explain:
            ranking = self.explain_recommendations(ranking)
        return ranking.reset_index(drop=True)

    def explain_recommendations(self, recommendations: pd.DataFrame) -> pd.DataFrame:
        """Acrescenta explicações baseadas nas contribuições TF-IDF."""
        result = recommendations.copy()
        if not result.empty and self._active_profile is not None:
            result["explanation"] = result.apply(self._build_explanation, axis=1)
        return result

    def _build_explanation(self, row: pd.Series) -> str:
        """Explica os termos TF-IDF com contribuição positiva para a pontuação."""
        recipe_id = recipe_id_key(row["RecipeId"])
        matrix_position = self.tfidf_builder.index_by_id[recipe_id]
        recipe_vector = self.tfidf_builder.matrix.getrow(matrix_position)
        contributions = np.asarray(recipe_vector.toarray()).ravel() * self._active_profile
        top_positions = np.argsort(contributions)[::-1]
        terms = [
            (self.tfidf_builder.feature_names[index], float(contributions[index]))
            for index in top_positions
            if contributions[index] > 1e-12
        ][:5]
        if not terms:
            return (
                "A receita apresenta compatibilidade textual geral com o perfil; "
                "não houve termos individuais com contribuição positiva suficiente para destacar."
            )

        ingredient_tokens = {
            token
            for ingredient in coerce_ingredient_list(row.get("RecipeIngredientParts", []))
            for token in ingredient.split()
        }
        ingredient_terms = [term for term, _ in terms if term in ingredient_tokens]
        if ingredient_terms:
            explanation = (
                "Ingredientes que contribuíram para a compatibilidade: "
                + ", ".join(ingredient_terms)
                + "."
            )
        else:
            explanation = (
                "Termos do conteúdo da receita que contribuíram para a compatibilidade: "
                + ", ".join(term for term, _ in terms)
                + "."
            )

        history = self._ensure_user_ratings()
        if not history.empty and "Rating" in history.columns:
            positive_history = history[history["Rating"] >= 4]
            history_positions = [
                self.tfidf_builder.index_by_id.get(recipe_id_key(history_id))
                for history_id in positive_history["RecipeId"]
            ]
            valid = [position for position in history_positions if position is not None]
            if valid:
                history_vectors = self.tfidf_builder.matrix[valid]
                similarities = (history_vectors @ recipe_vector.T).toarray().ravel()
                best_index = int(np.argmax(similarities))
                if similarities[best_index] > 0:
                    best_history_id = positive_history.iloc[
                        [index for index, position in enumerate(history_positions) if position is not None][best_index]
                    ]["RecipeId"]
                    history_names = self.recipes_df.set_index("RecipeId")["Name"] if "Name" in self.recipes_df else None
                    if history_names is not None and best_history_id in history_names.index:
                        explanation += f" A receita também se aproxima de “{history_names.loc[best_history_id]}”, bem avaliada no histórico."
        return explanation

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

        user_vector, _ = build_user_profile(self._ensure_user_ratings(), self.recipes_df, self.tfidf_builder)
        if user_vector is None:
            return pd.DataFrame(columns=["RecipeId", "score", "explanation"])

        ranked = self.compute_scores(user_vector, candidate_df=candidates, explain=not self.use_mmr)
        if self.use_mmr:
            ranked = self.mmr_diversify(ranked, self.diversity_lambda)
        output = ranked.head(top_n if top_n is not None else self.top_n).copy()
        if self.use_mmr:
            output["explanation"] = output.apply(self._build_explanation, axis=1)
        output["rank"] = range(1, len(output) + 1)
        return output.reset_index(drop=True)

    def mmr_diversify(self, ranked_df: pd.DataFrame, lambda_param: float | None = None) -> pd.DataFrame:
        """Reordena as recomendações usando MMR para equilibrar relevância e diversidade."""
        if ranked_df.empty or lambda_param is None:
            return ranked_df.copy()
        if not 0 <= lambda_param <= 1:
            raise ValueError("lambda_param deve estar entre 0 e 1.")

        remaining = ranked_df.reset_index(drop=True).copy()
        positions = [
            self.tfidf_builder.index_by_id.get(recipe_id_key(recipe_id))
            for recipe_id in remaining["RecipeId"]
        ]
        valid = [index for index, position in enumerate(positions) if position is not None]
        if len(valid) != len(remaining):
            remaining = remaining.iloc[valid].reset_index(drop=True)
            positions = [positions[index] for index in valid]
        vectors = self.tfidf_builder.matrix[positions]

        remaining_indices = list(range(len(remaining)))
        selected_indices: list[int] = []
        while remaining_indices:
            if not selected_indices:
                next_index = max(remaining_indices, key=lambda index: float(remaining.iloc[index]["score"]))
            else:
                selected_vectors = vectors[selected_indices]
                relevance = remaining["score"].to_numpy(dtype=float)
                diversity = (vectors[remaining_indices] @ selected_vectors.T).toarray().max(axis=1)
                mmr_scores = (
                    lambda_param * relevance[remaining_indices]
                    - (1 - lambda_param) * diversity
                )
                next_index = remaining_indices[int(np.argmax(mmr_scores))]
            selected_indices.append(next_index)
            remaining_indices.remove(next_index)

        result = remaining.iloc[selected_indices].copy().reset_index(drop=True)
        result["rank"] = np.arange(1, len(result) + 1)
        return result


__all__ = ["RecipeRecommender"]
