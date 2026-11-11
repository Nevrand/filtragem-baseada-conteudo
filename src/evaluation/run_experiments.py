from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.load_data import load_datasets
from src.data.preprocess import prepare_recipe_dataframe
from src.evaluation.baselines import popularity_recommendation, random_recommendation
from src.evaluation.evaluate import METRIC_COLUMNS, evaluate_user_recommendations, summarize_user_results
from src.evaluation.split import split_reviews_by_user
from src.features.text_features import build_tfidf_matrix, recipe_id_key
from src.features.user_profile import build_user_profile


def _recommend_content(builder, profile: np.ndarray | None, candidates: pd.DataFrame, top_n: int) -> list:
    if profile is None or candidates.empty:
        return []
    positions = [
        builder.index_by_id.get(recipe_id_key(recipe_id))
        for recipe_id in candidates["RecipeId"]
    ]
    valid = [(row_index, position) for row_index, position in enumerate(positions) if position is not None]
    if not valid:
        return []
    row_indices, matrix_positions = zip(*valid)
    scores = np.asarray(builder.matrix[list(matrix_positions)] @ profile).ravel()
    count = min(top_n, len(scores))
    selected = np.argpartition(scores, -count)[-count:] if count else np.array([], dtype=int)
    selected = selected[np.argsort(-scores[selected], kind="stable")]
    return candidates.iloc[[row_indices[index] for index in selected]]["RecipeId"].tolist()


def run_experiments(
    recipe_path: str | Path | None = None,
    review_path: str | Path | None = None,
    test_fraction: float = 0.2,
    random_state: int = 42,
    min_train_reviews: int = 1,
    top_n: int = 10,
    output_dir: str | Path = "results",
    max_users: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Executa avaliação por usuário sem usar avaliações de teste nos perfis."""
    if top_n < 10:
        raise ValueError("top_n deve ser pelo menos 10 para calcular as métricas @10.")
    recipes, reviews = load_datasets(
        recipe_path=recipe_path,
        review_path=review_path,
        recipe_required=("RecipeId", "Name", "Description", "RecipeCategory", "Keywords", "RecipeIngredientParts"),
        review_required=("RecipeId", "AuthorId", "Rating"),
    )
    if recipes.empty or reviews.empty:
        raise FileNotFoundError("Os arquivos recipes.parquet e reviews.parquet são necessários para os experimentos.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    from src.analysis.dataset_report import build_dataset_report

    dataset_summary = build_dataset_report(recipes, reviews, output_path)
    recipes = prepare_recipe_dataframe(recipes)
    reviews = reviews.dropna(subset=["AuthorId", "RecipeId", "Rating"]).copy()
    reviews["Rating"] = pd.to_numeric(reviews["Rating"], errors="coerce")
    reviews = reviews.dropna(subset=["Rating"])

    date_column = "DateSubmitted" if "DateSubmitted" in reviews.columns else None
    valid_dates = (
        date_column is not None
        and pd.to_datetime(
            reviews[date_column], errors="coerce", utc=True, format="mixed"
        ).notna().all()
    )
    train_reviews, test_reviews = split_reviews_by_user(
        reviews,
        test_fraction=test_fraction,
        random_state=random_state,
        date_column=date_column,
        min_train_reviews=min_train_reviews,
    )

    all_user_ids = sorted(reviews["AuthorId"].dropna().unique().tolist())
    user_ids = all_user_ids
    unselected_users = 0
    if max_users is not None:
        if max_users < 1:
            raise ValueError("max_users deve ser maior que zero.")
        training_eligible_ids = reviews.groupby("AuthorId").size()
        training_eligible_ids = sorted(training_eligible_ids[training_eligible_ids >= min_train_reviews + 1].index.tolist())
        rng = np.random.default_rng(random_state)
        user_ids = sorted(
            rng.choice(training_eligible_ids, size=min(max_users, len(training_eligible_ids)), replace=False).tolist()
        )
        unselected_users = len(all_user_ids) - len(user_ids)
    selected_train = train_reviews[train_reviews["AuthorId"].isin(user_ids)]
    selected_test = test_reviews[test_reviews["AuthorId"].isin(user_ids)]
    train_by_user = {user_id: group for user_id, group in selected_train.groupby("AuthorId", sort=False)}
    test_by_user = {user_id: group for user_id, group in selected_test.groupby("AuthorId", sort=False)}

    builder = build_tfidf_matrix(recipes, max_features=5000)
    outputs: list[dict[str, Any]] = []
    catalog_ids = set(recipes["RecipeId"].dropna().tolist())
    seen_by_all_evaluable_users: set | None = None
    train_only_count = 0

    for user_id in user_ids:
        user_train = train_by_user.get(user_id, reviews.iloc[:0])
        user_test = test_by_user.get(user_id, reviews.iloc[:0])
        if user_test.empty:
            train_only_count += 1
            exclusion_reason = (
                "insufficient_training_history"
                if len(user_train) < min_train_reviews + 1
                else "no_test_split"
            )
            for model in ("RecipeMatch-TF-IDF", "Aleatório", "Popularidade"):
                result = evaluate_user_recommendations(user_id, test_reviews, [], k=top_n)
                result.update({
                    "evaluated": False,
                    "exclusion_reason": exclusion_reason,
                    "model": model,
                    "n_train_reviews": int(len(user_train)),
                    "eligible_candidates": int(recipes["RecipeId"].nunique() - user_train["RecipeId"].nunique()),
                })
                outputs.append(result)
            continue

        relevant = set(user_test.loc[user_test["Rating"] >= 4, "RecipeId"].tolist())
        profile, _ = build_user_profile(user_train, recipes, builder)
        seen_ids = set(user_train["RecipeId"].dropna().tolist())
        candidates = recipes[~recipes["RecipeId"].isin(seen_ids)]
        relevant_candidates = relevant.intersection(set(candidates["RecipeId"].tolist()))

        if profile is None:
            reason = "no_informative_profile"
            content_recommendations: list = []
        else:
            reason = None
            if relevant:
                seen_by_all_evaluable_users = (
                    seen_ids
                    if seen_by_all_evaluable_users is None
                    else seen_by_all_evaluable_users.intersection(seen_ids)
                )
            content_recommendations = _recommend_content(builder, profile, candidates, top_n)
        random_recs = random_recommendation(
            candidates,
            user_id=user_id,
            top_n=top_n,
            random_state=random_state,
            train_reviews=user_train,
        )
        popular_recs = popularity_recommendation(
            train_reviews,
            candidates,
            top_n=top_n,
            train_reviews=user_train,
        )

        for model, recommendation_list in (
            ("RecipeMatch-TF-IDF", content_recommendations),
            ("Aleatório", random_recs),
            ("Popularidade", popular_recs),
        ):
            result = evaluate_user_recommendations(
                user_id,
                test_reviews,
                recommendation_list,
                k=top_n,
            )
            if reason:
                result["evaluated"] = False
                result["exclusion_reason"] = reason
                result.update({metric: 0.0 for metric in METRIC_COLUMNS})
            result["model"] = model
            result["n_train_reviews"] = int(len(user_train))
            result["eligible_candidates"] = int(len(candidates))
            result["n_test_relevant_candidates"] = len(relevant_candidates)
            outputs.append(result)

    by_user = pd.DataFrame(outputs)
    eligible_catalog_size = (
        len(catalog_ids - seen_by_all_evaluable_users)
        if seen_by_all_evaluable_users is not None
        else 0
    )
    summary_rows = []
    for model in ("RecipeMatch-TF-IDF", "Aleatório", "Popularidade"):
        model_results = by_user[by_user["model"] == model] if not by_user.empty else by_user
        summary = summarize_user_results(
            model_results,
            total_eligible_items=eligible_catalog_size,
            total_users_considered=len(user_ids),
        )
        summary["model"] = model
        summary["users_without_test_split"] = train_only_count
        summary_rows.append(summary)
    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(output_path / "metrics_summary.csv", index=False)
    by_user.to_csv(output_path / "metrics_by_user.csv", index=False)

    config = {
        "recipe_path": str(recipe_path) if recipe_path is not None else "data/raw/recipes.parquet",
        "review_path": str(review_path) if review_path is not None else "data/raw/reviews.parquet",
        "test_fraction": test_fraction,
        "random_state": random_state,
        "min_train_reviews": min_train_reviews,
        "top_n": top_n,
        "max_users": max_users,
        "user_sampling": "seeded random sample from distinct user IDs" if max_users is not None else "all distinct user IDs",
        "users_selected": len(user_ids),
        "user_population_size": len(all_user_ids),
        "users_not_in_sample": unselected_users,
        "evaluation_date_column": date_column,
        "chronological_split": bool(valid_dates),
        "relevance_threshold": 4,
        "rating_values": sorted(reviews["Rating"].dropna().unique().tolist()),
        "tfidf_max_features": 5000,
        "models": ["RecipeMatch-TF-IDF", "Aleatório", "Popularidade"],
        "coverage_denominator": "união das receitas candidatas elegíveis nos usuários avaliáveis, após excluir itens vistos no treino",
        "users_without_test_split": train_only_count,
    }
    dataset_summary.update({
        "train_review_count": int(len(train_reviews)),
        "test_review_count": int(len(test_reviews)),
        "users_evaluated": int(by_user.loc[by_user.get("evaluated", pd.Series(dtype=bool)), "user_id"].nunique()) if not by_user.empty else 0,
        "users_excluded_no_relevant_items": int(
            by_user.loc[by_user.get("exclusion_reason", pd.Series(dtype=object)) == "no_relevant_items", "user_id"].nunique()
        ) if not by_user.empty else 0,
        "users_excluded_no_profile": int(
            by_user.loc[by_user.get("exclusion_reason", pd.Series(dtype=object)) == "no_informative_profile", "user_id"].nunique()
        ) if not by_user.empty else 0,
        "users_without_test_split": train_only_count,
        "eligible_catalog_size": eligible_catalog_size,
    })
    (output_path / "evaluation_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_df, by_user, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa os experimentos offline do RecipeMatch.")
    parser.add_argument("--recipes", type=Path, help="Caminho para recipes.parquet.")
    parser.add_argument("--reviews", type=Path, help="Caminho para reviews.parquet.")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-train-reviews", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-users", type=int, help="Limita explicitamente usuários para uma execução de teste.")
    args = parser.parse_args()
    summary, _, config = run_experiments(
        recipe_path=args.recipes,
        review_path=args.reviews,
        test_fraction=args.test_fraction,
        random_state=args.random_state,
        min_train_reviews=args.min_train_reviews,
        top_n=args.top_n,
        output_dir=args.output_dir,
        max_users=args.max_users,
    )
    print(summary.to_string(index=False))
    print(f"Resultados exportados para {args.output_dir.resolve()}. Usuários selecionados: {config['users_selected']}.")


if __name__ == "__main__":
    main()
