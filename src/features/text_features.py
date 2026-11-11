from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.recommendation.similarity import cosine_similarity


@dataclass
class TextFeatureBuilder:
    """Empacota a criação e reutilização do espaço TF-IDF das receitas."""

    recipe_id_column: str = "RecipeId"
    text_column: str = "combined_text"
    min_df: float = 1.0
    max_features: int | None = 5000
    vectorizer: TfidfVectorizer | None = None
    matrix: sparse.csr_matrix | None = None
    recipe_ids: list[Any] | None = None
    feature_names: list[str] | None = None
    index_by_id: dict[str, int] | None = None

    def fit(self, recipes_df: pd.DataFrame, text_field: str | None = None) -> "TextFeatureBuilder":
        """Constrói uma matriz TF-IDF para todas as receitas do dataframe."""
        if recipes_df.empty:
            self.matrix = sparse.csr_matrix((0, 0))
            self.recipe_ids = []
            self.feature_names = []
            self.index_by_id = {}
            return self

        working_df = recipes_df.copy()
        if text_field is not None:
            working_df[self.text_column] = working_df[text_field]
        elif self.text_column not in working_df.columns:
            raise ValueError(f"Coluna de texto '{self.text_column}' não encontrada no dataframe.")

        min_df_value = 1 if self.min_df <= 1 and self.min_df > 0 else self.min_df
        self.vectorizer = TfidfVectorizer(
            min_df=min_df_value,
            max_features=self.max_features,
            lowercase=True,
            strip_accents="unicode",
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.matrix = self.vectorizer.fit_transform(working_df[self.text_column].fillna("").astype(str))
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        self.recipe_ids = working_df[self.recipe_id_column].astype(str).tolist()
        self.index_by_id = {str(recipe_id): idx for idx, recipe_id in enumerate(self.recipe_ids)}
        return self

    def transform(self, recipes_df: pd.DataFrame, text_field: str | None = None) -> sparse.csr_matrix:
        """Aplica o vetorizador já treinado em novas receitas."""
        if self.vectorizer is None:
            raise ValueError("O vetorizador TF-IDF ainda não foi treinado. Chame fit() antes.")
        working_df = recipes_df.copy()
        if text_field is not None:
            texts = working_df[text_field].fillna("").astype(str)
        else:
            texts = working_df[self.text_column].fillna("").astype(str)
        return self.vectorizer.transform(texts)

    def get_vector_for_recipe(self, recipe_id: Any) -> np.ndarray:
        """Retorna o vetor TF-IDF de uma receita específica."""
        if self.matrix is None or self.index_by_id is None:
            raise ValueError("A matriz TF-IDF ainda não foi construída.")
        key = str(recipe_id)
        if key not in self.index_by_id:
            raise KeyError(f"Receita {recipe_id} não foi encontrada no índice TF-IDF.")
        return self.matrix[self.index_by_id[key]].toarray().ravel()

    def get_top_terms(self, recipe_id: Any, top_k: int = 5) -> list[tuple[str, float]]:
        """Lista os termos mais relevantes para uma receita a partir do vetor TF-IDF."""
        if self.matrix is None or self.feature_names is None:
            raise ValueError("A matriz TF-IDF ainda não foi construída.")
        vector = self.get_vector_for_recipe(recipe_id)
        top_idx = np.argsort(vector)[::-1][:top_k]
        return [(self.feature_names[i], float(vector[i])) for i in top_idx if vector[i] > 0.0]

    def get_recipe_scores(self, user_vector: np.ndarray) -> np.ndarray:
        """Calcula a similaridade de cosseno entre o perfil do usuário e as receitas."""
        if self.matrix is None:
            raise ValueError("A matriz TF-IDF ainda não foi construída.")
        return cosine_similarity(user_vector, self.matrix).ravel()


def build_tfidf_matrix(
    recipes_df: pd.DataFrame,
    recipe_id_column: str = "RecipeId",
    text_field: str = "combined_text",
    min_df: float = 1.0,
    max_features: int | None = 5000,
) -> TextFeatureBuilder:
    """Função utilitária para construir rapidamente um vetorizador TF-IDF."""
    builder = TextFeatureBuilder(
        recipe_id_column=recipe_id_column,
        text_column=text_field,
        min_df=min_df,
        max_features=max_features,
    )
    return builder.fit(recipes_df, text_field=text_field)


__all__ = ["TextFeatureBuilder", "build_tfidf_matrix"]
