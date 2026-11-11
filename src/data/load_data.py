from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RECIPE_PATH = ROOT_DIR / "data" / "raw" / "recipes.parquet"
DEFAULT_REVIEW_PATH = ROOT_DIR / "data" / "raw" / "reviews.parquet"


def _resolve_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


def _validate_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}. Colunas disponíveis: {list(frame.columns)}")


def _available_columns(path: Path, requested: tuple[str, ...] | None) -> list[str] | None:
    if requested is None:
        return None
    import pyarrow.parquet as parquet

    available = set(parquet.read_schema(path).names)
    return [column for column in requested if column in available]


@lru_cache(maxsize=1)
def load_datasets(
    recipe_path: str | Path | None = None,
    review_path: str | Path | None = None,
    recipe_required: tuple[str, ...] = (
        "RecipeId",
        "Name",
        "Description",
        "RecipeCategory",
        "Keywords",
        "RecipeIngredientParts",
        "TotalTime",
    ),
    review_required: tuple[str, ...] = ("RecipeId", "AuthorId", "Rating"),
    recipe_columns: tuple[str, ...] | None = None,
    review_columns: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega receitas e avaliações do dataset Parquet.

    Se algum arquivo não estiver presente, a função informa claramente a limitação e
    retorna um DataFrame vazio para o conjunto ausente. Isso evita falhas silenciosas
    em execuções de demonstração ou em testes com fixtures leves.
    """
    recipe_file = _resolve_path(recipe_path, DEFAULT_RECIPE_PATH)
    review_file = _resolve_path(review_path, DEFAULT_REVIEW_PATH)

    recipes = pd.DataFrame()
    reviews = pd.DataFrame()

    if recipe_file.exists():
        recipes = pd.read_parquet(recipe_file, columns=_available_columns(recipe_file, recipe_columns))
        _validate_columns(recipes, recipe_required)
    else:
        print(f"Arquivo de receitas não encontrado: {recipe_file}")

    if review_file.exists():
        reviews = pd.read_parquet(review_file, columns=_available_columns(review_file, review_columns))
        _validate_columns(reviews, review_required)
    else:
        print(f"Arquivo de avaliações não encontrado: {review_file}")

    if not recipes.empty:
        recipes = recipes.copy()
        if "RecipeId" in recipes.columns:
            recipes["RecipeId"] = pd.to_numeric(recipes["RecipeId"], errors="coerce")
    if not reviews.empty:
        reviews = reviews.copy()
        for column in ("RecipeId", "AuthorId", "Rating"):
            if column in reviews.columns:
                reviews[column] = pd.to_numeric(reviews[column], errors="coerce")

    return recipes, reviews


def load_recipe_data(path: str | Path | None = None, required_columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Carrega apenas receitas e valida colunas essenciais."""
    dataset_path = _resolve_path(path, DEFAULT_RECIPE_PATH)
    if not dataset_path.exists():
        print(f"Arquivo de receitas não encontrado: {dataset_path}")
        return pd.DataFrame()

    recipe_df = pd.read_parquet(dataset_path)
    required = list(required_columns or ["RecipeId", "Name", "Description", "RecipeCategory", "Keywords", "RecipeIngredientParts", "TotalTime"])
    _validate_columns(recipe_df, required)
    return recipe_df.copy()


def load_review_data(path: str | Path | None = None, required_columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Carrega apenas avaliações e valida colunas essenciais."""
    dataset_path = _resolve_path(path, DEFAULT_REVIEW_PATH)
    if not dataset_path.exists():
        print(f"Arquivo de avaliações não encontrado: {dataset_path}")
        return pd.DataFrame()

    review_df = pd.read_parquet(dataset_path)
    required = list(required_columns or ["RecipeId", "AuthorId", "Rating"])
    _validate_columns(review_df, required)
    return review_df.copy()


def check_dataset_integrity(recipes: pd.DataFrame, reviews: pd.DataFrame) -> dict:
    """Retorna resumo rápido sobre duplicatas, dados ausentes e tamanho das tabelas."""
    summary = {
        "recipes_rows": len(recipes),
        "reviews_rows": len(reviews),
        "recipes_duplicates": int(recipes.duplicated(subset=["RecipeId"]).sum()) if "RecipeId" in recipes.columns else 0,
        "reviews_duplicates": int(reviews.duplicated(subset=["ReviewId"]).sum()) if "ReviewId" in reviews.columns else 0,
        "recipes_missing": recipes.isna().sum().to_dict() if not recipes.empty else {},
        "reviews_missing": reviews.isna().sum().to_dict() if not reviews.empty else {},
    }
    return summary


__all__ = [
    "DEFAULT_RECIPE_PATH",
    "DEFAULT_REVIEW_PATH",
    "load_datasets",
    "load_recipe_data",
    "load_review_data",
    "check_dataset_integrity",
]
