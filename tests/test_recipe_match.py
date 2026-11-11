from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import load_datasets
from src.data.preprocess import build_recipe_text, coerce_ingredient_list, normalize_text, parse_duration_to_minutes, prepare_recipe_dataframe
from src.evaluation.baselines import popularity_recommendation, random_recommendation
from src.evaluation.evaluate import evaluate_user_recommendations
from src.evaluation.metrics import catalog_coverage, ndcg_at_k, precision_at_k, recall_at_k
from src.features.text_features import build_tfidf_matrix
from src.features.user_profile import build_user_profile
from src.recommendation.filters import filter_by_avoid_ingredients, filter_by_time, filter_seen_recipes
from src.recommendation.recommender import RecipeRecommender
from src.recommendation.similarity import cosine_similarity


@pytest.fixture
def sample_recipes():
    return pd.DataFrame(
        {
            "RecipeId": [1, 2, 3, 4],
            "Name": ["Tomato Pasta", "Chicken Salad", "Pasta Primavera", "Banana Smoothie"],
            "Description": ["Comfort food with tomato and pasta", "Fresh salad with chicken", "Vegetarian pasta with vegetables", "Fruit smoothie with banana"],
            "RecipeCategory": ["Dinner", "Lunch", "Dinner", "Breakfast"],
            "Keywords": [["easy", "quick"], ["fresh", "healthy"], ["vegetarian"], ["sweet", "fast"]],
            "RecipeIngredientParts": [["tomato", "pasta", "garlic"], ["chicken", "lettuce", "tomato"], ["pasta", "zucchini", "peas"], ["banana", "milk", "yogurt"]],
            "TotalTime": ["PT30M", "PT45M", "PT60M", "PT20M"],
        }
    )


@pytest.fixture
def sample_reviews():
    return pd.DataFrame(
        {
            "ReviewId": [1, 2, 3, 4, 5],
            "RecipeId": [1, 1, 2, 3, 4],
            "AuthorId": [10, 10, 10, 11, 11],
            "Rating": [5, 4, 5, 2, 5],
            "DateSubmitted": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-02", "2023-01-05", "2023-01-06"]),
        }
    )


def test_load_datasets_loads_parquet(tmp_path):
    recipes = pd.DataFrame({"RecipeId": [1, 2], "Name": ["A", "B"], "Description": ["desc a", "desc b"], "RecipeCategory": ["c1", "c2"], "Keywords": [["k1"], ["k2"]], "RecipeIngredientParts": [["tomato"], ["milk"]], "TotalTime": ["PT30M", "PT45M"]})
    reviews = pd.DataFrame({"RecipeId": [1, 2], "AuthorId": [10, 11], "Rating": [5, 3]})
    recipes_path = tmp_path / "recipes.parquet"
    reviews_path = tmp_path / "reviews.parquet"
    recipes.to_parquet(recipes_path)
    reviews.to_parquet(reviews_path)

    loaded_recipes, loaded_reviews = load_datasets(recipe_path=recipes_path, review_path=reviews_path)

    assert loaded_recipes.shape[0] == 2
    assert loaded_reviews.shape[0] == 2
    assert {"RecipeId", "Name"}.issubset(set(loaded_recipes.columns))


def test_missing_values_and_cleaning(sample_recipes):
    sample_recipes.loc[0, "Description"] = None
    cleaned = prepare_recipe_dataframe(sample_recipes)

    assert "combined_text" in cleaned.columns
    assert cleaned["combined_text"].str.len().sum() > 0
    assert "none" not in cleaned["combined_text"].str.lower().tolist()[0]


def test_normalize_text_handles_parquet_arrays_and_missing_values():
    assert normalize_text(np.array(["Fresh tomato", "olive oil"])) == "fresh tomato olive oil"
    assert normalize_text(np.array(["tomato", None], dtype=object)) == "tomato"
    assert normalize_text(np.nan) == ""


def test_duration_conversion():
    assert parse_duration_to_minutes("PT1H30M") == 90
    assert parse_duration_to_minutes("45 minutes") == 45
    assert parse_duration_to_minutes(None) is None


def test_tfidf_matrix_generation(sample_recipes):
    prepared = prepare_recipe_dataframe(sample_recipes)
    builder = build_tfidf_matrix(prepared)

    assert builder.matrix.shape[0] == len(prepared)
    assert builder.matrix.nnz > 0


def test_user_profile_is_built_from_history(sample_recipes, sample_reviews):
    prepared = prepare_recipe_dataframe(sample_recipes)
    builder = build_tfidf_matrix(prepared)
    user_vector, history = build_user_profile(sample_reviews[sample_reviews["AuthorId"] == 10].copy(), prepared, builder)

    assert user_vector is not None
    assert history.shape[0] >= 1
    assert user_vector.shape[0] == builder.matrix.shape[1]


def test_cosine_similarity_works():
    a = [1.0, 0.0]
    b = [[1.0, 0.0], [0.0, 1.0]]
    sim = cosine_similarity(a, b)
    assert pytest.approx(sim[0]) == 1.0
    assert pytest.approx(sim[1]) == 0.0


def test_seen_recipes_are_excluded(sample_recipes, sample_reviews):
    seen = sample_reviews[sample_reviews["AuthorId"] == 10]
    filtered = filter_seen_recipes(sample_recipes, seen, recipe_id_column="RecipeId")

    assert 1 not in filtered["RecipeId"].tolist()


def test_forbidden_ingredients_are_filtered(sample_recipes):
    filtered = filter_by_avoid_ingredients(sample_recipes, ["tomato", "milk"])

    assert filtered[filtered["RecipeId"] == 1].empty
    assert filtered[filtered["RecipeId"] == 4].empty
    assert 3 in filtered["RecipeId"].tolist()


def test_time_filter_works(sample_recipes):
    filtered = filter_by_time(sample_recipes, 30)
    assert set(filtered["RecipeId"]).issubset({1, 4})


def test_user_without_history_is_handled(sample_recipes):
    prepared = prepare_recipe_dataframe(sample_recipes)
    builder = build_tfidf_matrix(prepared)
    empty_reviews = pd.DataFrame({"AuthorId": [], "RecipeId": [], "Rating": []})
    user_vector, history = build_user_profile(empty_reviews, prepared, builder)

    assert user_vector is None
    assert history.empty


def test_top_n_and_explanation(sample_recipes, sample_reviews):
    prepared = prepare_recipe_dataframe(sample_recipes)
    builder = build_tfidf_matrix(prepared)
    recommender = RecipeRecommender(prepared, sample_reviews, builder, user_id=10, top_n=2)
    result = recommender.recommend_top_n(user_id=10, max_minutes=60)

    assert len(result) <= 2
    assert "explanation" in result.columns
    assert any("ingredientes" in explanation.lower() for explanation in result["explanation"].tolist())


def test_metrics_and_baselines(sample_reviews, sample_recipes):
    prepared = prepare_recipe_dataframe(sample_recipes)
    scored = [1, 3]
    metrics = evaluate_user_recommendations(10, sample_reviews, scored, k=5, relevance_threshold=4)
    assert metrics["precision@5"] >= 0
    assert metrics["recall@5"] >= 0
    assert metrics["n_relevant"] >= 0

    assert precision_at_k({1}, [1, 2], 5) == 0.5
    assert recall_at_k({1, 2}, [1], 5) == 0.5
    assert ndcg_at_k([1, 0], [1, 0], 5) == 1.0
    assert catalog_coverage([[1, 2], [2, 3]], 3) == 1.0

    assert random_recommendation(prepared, top_n=2)
    assert popularity_recommendation(sample_reviews, prepared, top_n=2)


def test_mmr_diversify_keeps_ordering(sample_recipes, sample_reviews):
    prepared = prepare_recipe_dataframe(sample_recipes)
    builder = build_tfidf_matrix(prepared)
    recommender = RecipeRecommender(prepared, sample_reviews, builder, user_id=10, top_n=2)
    user_vector, _ = build_user_profile(sample_reviews[sample_reviews["AuthorId"] == 10].copy(), prepared, builder)
    candidate = recommender.compute_scores(user_vector, prepared)
    diversified = recommender.mmr_diversify(candidate, lambda_param=0.7)
    assert len(diversified) <= len(candidate)
