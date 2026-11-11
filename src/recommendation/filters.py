from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from src.data.preprocess import parse_duration_to_minutes


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
    forbidden = {str(item).strip().lower() for item in prohibited_ingredients if str(item).strip()}
    if not forbidden:
        return candidate_df.copy()

    def has_forbidden(row):
        ingredients = row.get(ingredient_column, [])
        if isinstance(ingredients, str):
            ingredients = [ingredients]
        normalized = {str(item).strip().lower() for item in ingredients if str(item).strip()}
        return bool(normalized.intersection(forbidden))

    return candidate_df[~candidate_df.apply(has_forbidden, axis=1)].copy()


def filter_by_time(candidate_df: pd.DataFrame, max_minutes: float | int | None, time_column: str = "TotalTime", fallback_column: str | None = None) -> pd.DataFrame:
    """Remove receitas acima do tempo máximo.

    Quando a coluna TotalTime não estiver disponível, usa fallback_column como alternativa.
    """
    if candidate_df.empty or max_minutes is None:
        return candidate_df.copy()

    if time_column not in candidate_df.columns:
        if fallback_column is None or fallback_column not in candidate_df.columns:
            return candidate_df.copy()
        time_column = fallback_column

    def safe_minutes(row):
        value = row.get(time_column)
        minutes = parse_duration_to_minutes(value)
        if minutes is None:
            return None
        return minutes

    candidate_df = candidate_df.copy()
    candidate_df["_minutes"] = candidate_df.apply(safe_minutes, axis=1)
    candidate_df = candidate_df[candidate_df["_minutes"].isna() | (candidate_df["_minutes"] <= float(max_minutes))].copy()
    return candidate_df.drop(columns=["_minutes"], errors="ignore")


def filter_by_available_ingredients(candidate_df: pd.DataFrame, available_ingredients: Sequence[str] | None, ingredient_column: str = "RecipeIngredientParts") -> pd.DataFrame:
    """Opção extra: mantém receitas que usam pelo menos um ingrediente disponível em casa."""
    if candidate_df.empty or not available_ingredients:
        return candidate_df.copy()
    wanted = {str(item).strip().lower() for item in available_ingredients if str(item).strip()}
    if not wanted:
        return candidate_df.copy()

    def contains_any(row):
        ingredients = row.get(ingredient_column, [])
        if isinstance(ingredients, str):
            ingredients = [ingredients]
        normalized = {str(item).strip().lower() for item in ingredients if str(item).strip()}
        return bool(normalized.intersection(wanted))

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
