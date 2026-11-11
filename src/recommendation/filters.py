from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.data.preprocess import coerce_ingredient_list, normalize_text, parse_duration_to_minutes


def _ingredient_terms(value) -> list[tuple[str, ...]]:
    """Converte ingredientes em sequências de palavras normalizadas."""
    return [
        tuple(normalized.split())
        for item in coerce_ingredient_list(value)
        if (normalized := normalize_text(item))
    ]


def _contains_ingredient(ingredients, query_terms: set[tuple[str, ...]]) -> bool:
    for ingredient in _ingredient_terms(ingredients):
        for query in query_terms:
            if not query:
                continue
            width = len(query)
            if any(ingredient[index:index + width] == query for index in range(len(ingredient) - width + 1)):
                return True
    return False


def _query_terms(values: Sequence[str] | None) -> set[tuple[str, ...]]:
    return {
        tuple(normalized.split())
        for value in (values or [])
        if (normalized := normalize_text(value))
    }


def filter_seen_recipes(candidate_df: pd.DataFrame, user_ratings: pd.DataFrame | None, recipe_id_column: str = "RecipeId") -> pd.DataFrame:
    """Remove receitas já avaliadas pelo usuário."""
    if candidate_df.empty:
        return candidate_df.copy()
    if user_ratings is None or user_ratings.empty:
        return candidate_df.copy()
    seen = set(user_ratings[recipe_id_column].dropna().astype(int).tolist())
    if not seen:
        return candidate_df.copy()
    return candidate_df[~candidate_df[recipe_id_column].astype(int).isin(seen)].copy()


def filter_by_avoid_ingredients(candidate_df: pd.DataFrame, prohibited_ingredients: Sequence[str] | None, ingredient_column: str = "RecipeIngredientParts") -> pd.DataFrame:
    """Remove receitas com ingredientes proibidos pelo usuário."""
    if candidate_df.empty or not prohibited_ingredients:
        return candidate_df.copy()
    forbidden = _query_terms(prohibited_ingredients)
    if not forbidden:
        return candidate_df.copy()

    def has_forbidden(row):
        return _contains_ingredient(row.get(ingredient_column, []), forbidden)

    return candidate_df[~candidate_df.apply(has_forbidden, axis=1)].copy()


def filter_by_time(candidate_df: pd.DataFrame, max_minutes: float | int | None, time_column: str = "TotalTime", fallback_column: str | None = None) -> pd.DataFrame:
    """Remove receitas acima do limite e receitas sem tempo conhecido.

    ``TotalTime`` é prioritário. Para linhas sem esse valor, ``PrepTime`` e
    ``CookTime`` só são somados quando ambos puderem ser convertidos.
    """
    if candidate_df.empty or max_minutes is None:
        return candidate_df.copy()

    def safe_minutes(row):
        minutes = parse_duration_to_minutes(row.get(time_column)) if time_column in row.index else None
        if minutes is None and {"PrepTime", "CookTime"}.issubset(row.index):
            prep = parse_duration_to_minutes(row.get("PrepTime"))
            cook = parse_duration_to_minutes(row.get("CookTime"))
            if prep is not None and cook is not None:
                minutes = prep + cook
        if minutes is None and fallback_column and fallback_column in row.index:
            minutes = parse_duration_to_minutes(row.get(fallback_column))
        return minutes

    candidate_df = candidate_df.copy()
    candidate_df["_minutes"] = candidate_df.apply(safe_minutes, axis=1)
    candidate_df = candidate_df[candidate_df["_minutes"].notna() & (candidate_df["_minutes"] <= float(max_minutes))].copy()
    return candidate_df.drop(columns=["_minutes"], errors="ignore")


def filter_by_available_ingredients(candidate_df: pd.DataFrame, available_ingredients: Sequence[str] | None, ingredient_column: str = "RecipeIngredientParts") -> pd.DataFrame:
    """Opção extra: mantém receitas que usam pelo menos um ingrediente disponível em casa."""
    if candidate_df.empty or not available_ingredients:
        return candidate_df.copy()
    wanted = _query_terms(available_ingredients)
    if not wanted:
        return candidate_df.copy()

    def contains_any(row):
        return _contains_ingredient(row.get(ingredient_column, []), wanted)

    return candidate_df[candidate_df.apply(contains_any, axis=1)].copy()


def filter_candidates(
    candidates: pd.DataFrame,
    user_ratings: pd.DataFrame | None,
    max_minutes: float | int | None = None,
    avoided_ingredients: Sequence[str] | None = None,
    available_ingredients: Sequence[str] | None = None,
    ingredient_column: str = "RecipeIngredientParts",
    recipe_id_column: str = "RecipeId",
) -> pd.DataFrame:
    """Aplica todos os filtros relevantes antes de montar o ranking final."""
    filtered = candidates.copy()
    filtered = filter_seen_recipes(filtered, user_ratings, recipe_id_column=recipe_id_column)
    filtered = filter_by_avoid_ingredients(filtered, avoided_ingredients, ingredient_column=ingredient_column)
    filtered = filter_by_time(filtered, max_minutes=max_minutes)
    if available_ingredients:
        filtered = filter_by_available_ingredients(filtered, available_ingredients, ingredient_column=ingredient_column)
    return filtered


__all__ = [
    "filter_seen_recipes",
    "filter_by_avoid_ingredients",
    "filter_by_time",
    "filter_by_available_ingredients",
    "filter_candidates",
]
