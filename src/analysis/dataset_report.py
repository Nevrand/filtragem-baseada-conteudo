from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load_data import load_datasets
from src.data.preprocess import coerce_ingredient_list, parse_duration_to_minutes


def build_dataset_report(
    recipes: pd.DataFrame,
    reviews: pd.DataFrame,
    output_dir: str | Path = "results",
) -> dict[str, Any]:
    """Calcula e exporta estatísticas descritivas diretamente dos DataFrames."""
    if recipes.empty or reviews.empty:
        raise ValueError("Não é possível gerar relatório sem receitas e avaliações.")
    required_reviews = {"RecipeId", "AuthorId", "Rating"}
    if not required_reviews.issubset(reviews.columns):
        raise ValueError(f"Colunas de avaliações ausentes: {sorted(required_reviews - set(reviews.columns))}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_ratings = pd.to_numeric(reviews["Rating"], errors="coerce")
    valid_ratings = all_ratings[all_ratings.between(1, 5)]
    unrated_interactions = all_ratings.eq(0)
    user_counts = reviews.groupby("AuthorId", dropna=True).size()
    recipe_counts = reviews.groupby("RecipeId", dropna=True).size()
    ingredients_column = "RecipeIngredientParts"
    recipes_with_ingredients = (
        int(recipes[ingredients_column].map(lambda value: bool(coerce_ingredient_list(value))).sum())
        if ingredients_column in recipes.columns
        else 0
    )

    missing_rows = []
    for dataset_name, frame, columns in (
        ("recipes", recipes, ["RecipeId", "Name", "Description", "RecipeCategory", "Keywords", ingredients_column, "TotalTime"]),
        ("reviews", reviews, ["ReviewId", "RecipeId", "AuthorId", "Rating", "Review", "DateSubmitted"]),
    ):
        for column in columns:
            if column in frame.columns:
                count = int(frame[column].isna().sum())
                missing_rows.append({
                    "dataset": dataset_name,
                    "column": column,
                    "missing_count": count,
                    "missing_fraction": count / len(frame) if len(frame) else 0.0,
                })
    missing_values = pd.DataFrame(missing_rows)
    missing_values.to_csv(output_path / "dataset_missing_values.csv", index=False)

    rating_distribution = valid_ratings.value_counts().sort_index().rename_axis("rating").reset_index(name="count")
    rating_distribution["fraction"] = rating_distribution["count"] / max(len(valid_ratings), 1)
    rating_distribution.to_csv(output_path / "rating_distribution.csv", index=False)

    if "RecipeCategory" in recipes.columns:
        categories = recipes["RecipeCategory"].dropna().astype(str).str.strip()
        top_categories = categories[categories.ne("")].value_counts().head(20).rename_axis("category").reset_index(name="recipe_count")
    else:
        top_categories = pd.DataFrame(columns=["category", "recipe_count"])
    top_categories.to_csv(output_path / "top_categories.csv", index=False)

    durations = None
    duration_column = next(
        (column for column in ("TotalTime", "PrepTime", "CookTime") if column in recipes.columns),
        None,
    )
    if duration_column:
        durations = recipes[duration_column].map(parse_duration_to_minutes).dropna()
    time_summary = {
        "source_column": duration_column,
        "known_count": int(len(durations)) if durations is not None else 0,
        "missing_or_invalid_count": int(len(recipes) - len(durations)) if durations is not None else int(len(recipes)),
    }
    if durations is not None and not durations.empty:
        time_summary.update({
            "minimum_minutes": float(durations.min()),
            "median_minutes": float(durations.median()),
            "mean_minutes": float(durations.mean()),
            "p90_minutes": float(durations.quantile(0.9)),
            "maximum_minutes": float(durations.max()),
        })

    summary = {
        "recipe_count": int(recipes["RecipeId"].nunique()) if "RecipeId" in recipes.columns else int(len(recipes)),
        "interaction_count": int(len(reviews)),
        "review_count": int(len(reviews)),
        "valid_rating_count": int(len(valid_ratings)),
        "unrated_interaction_count": int(unrated_interactions.sum()),
        "invalid_rating_count": int((all_ratings.notna() & ~all_ratings.between(1, 5) & ~unrated_interactions).sum()),
        "distinct_users": int(reviews["AuthorId"].nunique()),
        "distinct_interacted_recipes": int(reviews["RecipeId"].nunique()),
        "distinct_rated_recipes": int(
            reviews.loc[all_ratings.between(1, 5), "RecipeId"].nunique()
        ),
        "rating_min": float(valid_ratings.min()) if not valid_ratings.empty else None,
        "rating_max": float(valid_ratings.max()) if not valid_ratings.empty else None,
        "rating_mean": float(valid_ratings.mean()) if not valid_ratings.empty else None,
        "rating_distribution": {
            str(key): int(value)
            for key, value in valid_ratings.value_counts().sort_index().items()
        },
        "reviews_per_user": {
            "minimum": int(user_counts.min()) if not user_counts.empty else 0,
            "median": float(user_counts.median()) if not user_counts.empty else 0.0,
            "mean": float(user_counts.mean()) if not user_counts.empty else 0.0,
            "maximum": int(user_counts.max()) if not user_counts.empty else 0,
        },
        "reviews_per_recipe": {
            "minimum": int(recipe_counts.min()) if not recipe_counts.empty else 0,
            "median": float(recipe_counts.median()) if not recipe_counts.empty else 0.0,
            "mean": float(recipe_counts.mean()) if not recipe_counts.empty else 0.0,
            "maximum": int(recipe_counts.max()) if not recipe_counts.empty else 0,
        },
        "recipes_with_ingredients": recipes_with_ingredients,
        "recipe_time_minutes": time_summary,
        "columns_present": {"recipes": list(recipes.columns), "reviews": list(reviews.columns)},
    }
    (output_path / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera um relatório estatístico do dataset Food.com.")
    parser.add_argument("--recipes", type=Path, help="Caminho para recipes.parquet.")
    parser.add_argument("--reviews", type=Path, help="Caminho para reviews.parquet.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    recipes, reviews = load_datasets(
        recipe_path=args.recipes,
        review_path=args.reviews,
        recipe_required=("RecipeId",),
        review_required=("RecipeId", "AuthorId", "Rating"),
    )
    summary = build_dataset_report(recipes, reviews, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
