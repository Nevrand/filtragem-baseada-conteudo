from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.load_data import load_datasets
from src.data.preprocess import coerce_ingredient_list, prepare_recipe_dataframe
from src.evaluation.split import split_reviews_by_user
from src.features.text_features import build_tfidf_matrix
from src.features.user_profile import build_user_profile
from src.recommendation.recommender import RecipeRecommender


def _format_history(reviews: pd.DataFrame, recipes: pd.DataFrame, positive: bool) -> str:
    threshold = reviews["Rating"] >= 4 if positive else reviews["Rating"] <= 2
    selected = reviews[threshold].merge(
        recipes[["RecipeId", "Name", "RecipeIngredientParts"]],
        on="RecipeId",
        how="left",
    )
    records = []
    for row in selected.itertuples(index=False):
        records.append({
            "recipe": getattr(row, "Name", None),
            "rating": float(row.Rating),
            "ingredients": coerce_ingredient_list(getattr(row, "RecipeIngredientParts", [])),
        })
    return json.dumps(records, ensure_ascii=False)


def run_qualitative_analysis(
    users: list[int] | None = None,
    recipe_path: str | Path | None = None,
    review_path: str | Path | None = None,
    output_path: str | Path = "results/qualitative_examples.csv",
    top_n: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Gera exemplos qualitativos usando históricos de treino e recomendações reais."""
    recipes, reviews = load_datasets(
        recipe_path=recipe_path,
        review_path=review_path,
        recipe_required=("RecipeId", "Name", "Description", "RecipeCategory", "Keywords", "RecipeIngredientParts"),
        review_required=("RecipeId", "AuthorId", "Rating"),
    )
    if recipes.empty or reviews.empty:
        raise FileNotFoundError("Os arquivos Parquet reais são necessários para análise qualitativa.")
    recipes = prepare_recipe_dataframe(recipes)
    if "DateSubmitted" in reviews.columns:
        reviews["DateSubmitted"] = pd.to_datetime(
            reviews["DateSubmitted"], errors="coerce", utc=True, format="mixed"
        )
    train_reviews, test_reviews = split_reviews_by_user(
        reviews,
        test_fraction=0.2,
        random_state=random_state,
        date_column="DateSubmitted" if "DateSubmitted" in reviews.columns else None,
        min_train_reviews=2,
    )
    if users is None:
        enough_training = set(train_reviews.groupby("AuthorId").size().loc[lambda counts: counts >= 2].index)
        positive_test_users = set(test_reviews.loc[test_reviews["Rating"] >= 4, "AuthorId"].unique())
        eligible_users = sorted(enough_training.intersection(positive_test_users))
        sampled_users = pd.Series(eligible_users).sample(
            n=min(3, len(eligible_users)),
            random_state=random_state,
        ).tolist()
        selected_users = sorted(sampled_users)
    else:
        if not 2 <= len(users) <= 3:
            raise ValueError("Informe dois ou três IDs de usuários.")
        unknown = set(users) - set(reviews["AuthorId"].unique())
        if unknown:
            raise ValueError(f"Usuários não encontrados no dataset: {sorted(unknown)}")
        selected_users = users

    builder = build_tfidf_matrix(recipes, max_features=5000)
    selected_train = train_reviews[train_reviews["AuthorId"].isin(selected_users)]
    selected_test = test_reviews[test_reviews["AuthorId"].isin(selected_users)]
    train_by_user = {user_id: group for user_id, group in selected_train.groupby("AuthorId", sort=False)}
    test_by_user = {user_id: group for user_id, group in selected_test.groupby("AuthorId", sort=False)}

    output_rows = []
    recipe_lookup = recipes.set_index("RecipeId")
    for user_id in selected_users:
        user_train = train_by_user.get(user_id, reviews.iloc[:0])
        user_test = test_by_user.get(user_id, reviews.iloc[:0])
        profile, _ = build_user_profile(user_train, recipes, builder)
        recommendation_frame = pd.DataFrame()
        if profile is not None:
            seen = set(user_train["RecipeId"].dropna())
            candidates = recipes[~recipes["RecipeId"].isin(seen)]
            recommender = RecipeRecommender(
                recipes_df=recipes,
                reviews_df=user_train,
                tfidf_builder=builder,
                user_id=user_id,
                top_n=top_n,
            )
            recommendation_frame = recommender.compute_scores(profile, candidates, top_n=top_n)

        relevant_test_ids = set(user_test.loc[user_test["Rating"] >= 4, "RecipeId"])
        recommended_ids = set(recommendation_frame["RecipeId"]) if not recommendation_frame.empty else set()
        missed = relevant_test_ids - recommended_ids
        if recommendation_frame.empty:
            limitation = "Nenhuma recomendação foi gerada para este perfil e catálogo elegível."
        elif missed:
            limitation = f"{len(missed)} receita(s) bem avaliadas no conjunto de teste não apareceram no Top-{top_n}."
        else:
            limitation = "Nenhuma limitação de recuperação observada neste holdout; isso não garante qualidade subjetiva."

        positive_history = _format_history(user_train, recipes, positive=True)
        negative_history = _format_history(user_train, recipes, positive=False)
        mean_rating = float(user_train["Rating"].mean()) if not user_train.empty else None
        if recommendation_frame.empty:
            output_rows.append({
                "user_id": user_id,
                "train_review_count": len(user_train),
                "mean_train_rating": mean_rating,
                "positive_history": positive_history,
                "negative_history": negative_history,
                "recipe_id": None,
                "recipe_name": None,
                "score": None,
                "explanation": None,
                "test_relevant_recipe_count": len(relevant_test_ids),
                "test_relevant_recommended": 0,
                "limitation": limitation,
            })
            continue

        for row in recommendation_frame.itertuples(index=False):
            recipe = recipe_lookup.loc[row.RecipeId]
            output_rows.append({
                "user_id": user_id,
                "train_review_count": len(user_train),
                "mean_train_rating": mean_rating,
                "positive_history": positive_history,
                "negative_history": negative_history,
                "recipe_id": row.RecipeId,
                "recipe_name": recipe.get("Name"),
                "score": float(row.score),
                "explanation": row.explanation,
                "test_relevant_recipe_count": len(relevant_test_ids),
                "test_relevant_recommended": int(row.RecipeId in relevant_test_ids),
                "limitation": limitation,
            })

    result = pd.DataFrame(output_rows)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera exemplos qualitativos do RecipeMatch.")
    parser.add_argument("--users", nargs="+", type=int, help="IDs de dois ou três usuários.")
    parser.add_argument("--recipes", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/qualitative_examples.csv"))
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    result = run_qualitative_analysis(
        users=args.users,
        recipe_path=args.recipes,
        review_path=args.reviews,
        output_path=args.output,
        top_n=args.top_n,
        random_state=args.random_state,
    )
    if result.empty:
        print("Nenhum usuário elegível para a análise qualitativa.")
    else:
        summary = result.groupby("user_id", as_index=False).agg(
            train_review_count=("train_review_count", "first"),
            test_relevant_recipe_count=("test_relevant_recipe_count", "first"),
            recommendations=("recipe_id", "count"),
            test_relevant_recommended=("test_relevant_recommended", "sum"),
            limitation=("limitation", "first"),
        )
        print(summary.to_string(index=False))
    print(f"Exemplos exportados para {args.output.resolve()}.")


if __name__ == "__main__":
    main()
