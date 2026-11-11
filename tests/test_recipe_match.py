from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from src.analysis.dataset_report import build_dataset_report
from src.data.load_data import load_datasets
from src.data.preprocess import build_recipe_text, coerce_ingredient_list, normalize_text, parse_duration_to_minutes, prepare_recipe_dataframe
from src.evaluation.baselines import popularity_recommendation, random_recommendation
from src.evaluation.evaluate import evaluate_user_recommendations, summarize_user_results
from src.evaluation.metrics import catalog_coverage, ndcg_at_k, precision_at_k, recall_at_k
from src.evaluation.qualitative_analysis import run_qualitative_analysis
from src.evaluation.run_experiments import run_experiments
from src.evaluation.split import split_reviews_by_user
from src.features.text_features import build_tfidf_matrix
from src.features.user_profile import build_user_profile
from src.recommendation.filters import filter_by_avoid_ingredients, filter_by_available_ingredients, filter_by_time, filter_seen_recipes
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

    selected_recipes, selected_reviews = load_datasets(
        recipe_path=recipes_path,
        review_path=reviews_path,
        recipe_required=("RecipeId", "Name"),
        review_required=("RecipeId", "Rating"),
        recipe_columns=("RecipeId", "Name", "Images"),
        review_columns=("RecipeId", "Rating", "DateSubmitted"),
    )
    assert selected_recipes.columns.tolist() == ["RecipeId", "Name"]
    assert selected_reviews.columns.tolist() == ["RecipeId", "Rating"]


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
    assert parse_duration_to_minutes("P1DT2H") == 1560
    assert parse_duration_to_minutes("PT45S") == 0.75
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

    assert precision_at_k({1}, [1, 2], 5) == 0.2
    assert recall_at_k({1, 2}, [1], 5) == 0.5
    assert ndcg_at_k({1, 2}, [1, 2], 5) == 1.0
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


class _VectorLookup:
    def __init__(self, vectors):
        self.vectors = vectors

    def get_vector_for_recipe(self, recipe_id):
        return np.asarray(self.vectors[recipe_id], dtype=float)


@pytest.mark.parametrize(
    ("rating", "expected_weight"),
    [(5, 2.0), (4, 1.0), (3, 0.0), (2, -1.0), (1, -2.0), (0, -2.0)],
)
def test_profile_uses_signed_rating_weights(rating, expected_weight):
    recipes = pd.DataFrame({"RecipeId": [1], "combined_text": ["tomato"]})
    reviews = pd.DataFrame({"AuthorId": [9], "RecipeId": [1], "Rating": [rating]})
    profile, history = build_user_profile(reviews, recipes, _VectorLookup({1: [1.0, 0.0]}))

    assert history["score_weight"].tolist() == [expected_weight]
    if expected_weight > 0:
        assert profile.tolist() == [1.0, 0.0]
    elif expected_weight < 0:
        assert profile.tolist() == [-1.0, 0.0]
    else:
        assert profile is None


def test_duplicate_user_recipe_ratings_are_averaged():
    recipes = pd.DataFrame({"RecipeId": [1], "combined_text": ["tomato"]})
    reviews = pd.DataFrame({"AuthorId": [9, 9], "RecipeId": [1, 1], "Rating": [1, 5]})
    profile, history = build_user_profile(reviews, recipes, _VectorLookup({1: [1.0, 0.0]}))

    assert profile is None
    assert history["Rating"].tolist() == [3.0]
    assert history["score_weight"].tolist() == [0.0]


def test_profile_cancels_to_none_when_positive_and_negative_vectors_cancel():
    recipes = pd.DataFrame({"RecipeId": [1, 2], "combined_text": ["tomato", "tomato"]})
    reviews = pd.DataFrame({"AuthorId": [9, 9], "RecipeId": [1, 2], "Rating": [5, 1]})
    profile, _ = build_user_profile(
        reviews,
        recipes,
        _VectorLookup({1: [1.0, 0.0], 2: [1.0, 0.0]}),
    )
    assert profile is None


def test_chronological_split_uses_latest_events_and_keeps_user_training():
    reviews = pd.DataFrame({
        "AuthorId": [1, 1, 1, 1, 2],
        "RecipeId": [1, 2, 3, 4, 5],
        "Rating": [5, 4, 3, 5, 5],
        "DateSubmitted": ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-04", "2024-01-01"],
    })
    train, test = split_reviews_by_user(reviews, test_fraction=0.25)
    user_one_train = pd.to_datetime(train.loc[train.AuthorId == 1, "DateSubmitted"])
    user_one_test = pd.to_datetime(test.loc[test.AuthorId == 1, "DateSubmitted"])

    assert user_one_train.max() < user_one_test.min()
    assert (train.groupby("AuthorId").size() >= 1).all()


def test_chronological_split_never_splits_equal_timestamps_between_train_and_test():
    reviews = pd.DataFrame({
        "AuthorId": [1, 1, 1, 1],
        "RecipeId": [1, 2, 3, 4],
        "Rating": [5, 4, 3, 5],
        "DateSubmitted": ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"],
    })
    train, test = split_reviews_by_user(reviews, test_fraction=0.5)
    train_dates = pd.to_datetime(train["DateSubmitted"])
    test_dates = pd.to_datetime(test["DateSubmitted"])

    assert train_dates.max() < test_dates.min()
    assert set(train["RecipeId"]).isdisjoint(test["RecipeId"])


def test_invalid_dates_fall_back_to_reproducible_random_split():
    reviews = pd.DataFrame({
        "AuthorId": [1] * 5,
        "RecipeId": [1, 2, 3, 4, 5],
        "Rating": [1, 2, 3, 4, 5],
        "DateSubmitted": ["bad", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
    })
    train_a, test_a = split_reviews_by_user(reviews, random_state=42)
    train_b, test_b = split_reviews_by_user(reviews, random_state=42)
    assert train_a["RecipeId"].tolist() == train_b["RecipeId"].tolist()
    assert test_a["RecipeId"].tolist() == test_b["RecipeId"].tolist()
    assert set(train_a.RecipeId).isdisjoint(test_a.RecipeId)


def test_metric_values_at_five_and_ten_are_exact():
    relevant = {1, 3, 5}
    recommendations = [3, 8, 1, 9, 5, 10, 11, 12, 13, 14]
    result = evaluate_user_recommendations(
        7,
        pd.DataFrame({"AuthorId": [7, 7, 7, 7], "RecipeId": [1, 3, 5, 8], "Rating": [5, 4, 5, 3]}),
        recommendations,
    )

    assert result["precision@5"] == 3 / 5
    assert result["recall@5"] == 1.0
    assert result["precision@10"] == 3 / 10
    assert result["recall@10"] == 1.0
    expected_ndcg5 = (1 + 1 / np.log2(4) + 1 / np.log2(6)) / (1 + 1 / np.log2(3) + 1 / np.log2(4))
    expected_ndcg10 = (1 + 1 / np.log2(4) + 1 / np.log2(6)) / (1 + 1 / np.log2(3) + 1 / np.log2(4))
    assert result["ndcg@5"] == pytest.approx(expected_ndcg5)
    assert result["ndcg@10"] == pytest.approx(expected_ndcg10)


def test_users_without_relevant_test_ratings_are_reported_but_excluded_from_mean():
    results = pd.DataFrame([
        {"user_id": 1, "evaluated": True, "exclusion_reason": None, "n_test_reviews": 2, "n_recommendations": 10, "recommended": [1], "precision@5": 0.2},
        {"user_id": 2, "evaluated": False, "exclusion_reason": "no_relevant_items", "n_test_reviews": 2, "n_recommendations": 10, "recommended": [2], "precision@5": 0.0},
    ])
    summary = summarize_user_results(results, total_eligible_items=20)
    assert summary["precision@5"] == 0.2
    assert summary["evaluated_users"] == 1
    assert summary["excluded_users"] == 1
    assert summary["excluded_reasons"] == {"no_relevant_items": 1}
    assert summary["coverage"] == 1 / 20


def test_baselines_use_only_candidates_and_exclude_training_history(sample_recipes):
    train = pd.DataFrame({"RecipeId": [1, 2, 2, 2, 3, 3]})
    candidates = sample_recipes.copy()
    random_ids = random_recommendation(candidates, user_id=20, top_n=4, random_state=42, train_reviews=train)
    repeat_random_ids = random_recommendation(candidates, user_id=20, top_n=4, random_state=42, train_reviews=train)
    popular_ids = popularity_recommendation(train, candidates, top_n=4, train_reviews=train)

    assert random_ids == repeat_random_ids
    assert not set(random_ids).intersection({1, 2, 3})
    assert popular_ids == [4]


def test_ingredient_filters_use_full_words_for_avoid_and_available():
    frame = pd.DataFrame({
        "RecipeId": [1, 2, 3],
        "RecipeIngredientParts": [["fresh tomato"], ["tomatillo"], ["chopped tomato"]],
    })
    avoided = filter_by_avoid_ingredients(frame, ["tomato"])
    available = filter_by_available_ingredients(frame, ["tomato"])
    assert avoided["RecipeId"].tolist() == [2]
    assert available["RecipeId"].tolist() == [1, 3]


def test_time_filter_drops_unknown_and_uses_prep_plus_cook():
    frame = pd.DataFrame({
        "RecipeId": [1, 2, 3],
        "TotalTime": [None, "invalid", "PT10M"],
        "PrepTime": ["PT10M", "PT5M", "PT5M"],
        "CookTime": ["PT20M", None, "PT5M"],
    })
    filtered = filter_by_time(frame, 30)
    assert filtered["RecipeId"].tolist() == [1, 3]


def test_explanation_terms_have_positive_tfidf_profile_contributions():
    frame = pd.DataFrame({
        "RecipeId": [1, 2, 3],
        "Name": ["Tomato pasta", "Tomato soup", "Banana smoothie"],
        "Description": ["tomato pasta", "tomato soup", "banana smoothie"],
        "RecipeCategory": ["dinner", "dinner", "breakfast"],
        "Keywords": [[], [], []],
        "RecipeIngredientParts": [["tomato", "pasta"], ["tomato", "soup"], ["banana", "milk"]],
    })
    prepared = prepare_recipe_dataframe(frame)
    builder = build_tfidf_matrix(prepared)
    training = pd.DataFrame({"AuthorId": [9], "RecipeId": [1], "Rating": [5]})
    recommender = RecipeRecommender(prepared, training, builder, user_id=9)
    profile, _ = build_user_profile(training, prepared, builder)
    result = recommender.compute_scores(profile, prepared[prepared.RecipeId == 2], top_n=1)

    assert "tomato" in result.iloc[0]["explanation"].lower()
    recipe_vector = builder.matrix[builder.index_by_id["2"]].toarray().ravel()
    contributions = recipe_vector * profile
    assert contributions[builder.feature_names.index("tomato")] > 0


def test_mmr_reorders_similar_items_and_returns_more_than_five(sample_recipes):
    recipes = pd.DataFrame({
        "RecipeId": [1, 2, 3, 4, 5, 6, 7],
        "combined_text": [
            "apple banana fruit", "apple banana fruit sweet", "carrot soup savory",
            "rice beans dinner", "lemon fish dinner", "chocolate cake dessert", "oat milk breakfast",
        ],
    })
    builder = build_tfidf_matrix(recipes)
    recommender = RecipeRecommender(recipes, pd.DataFrame(), builder)
    ranked = pd.DataFrame({"RecipeId": [1, 2, 3, 4, 5, 6, 7], "score": [0.95, 0.9, 0.7, 0.65, 0.6, 0.55, 0.5]})
    diversified = recommender.mmr_diversify(ranked, lambda_param=0.5)

    assert len(diversified) == 7
    assert diversified.iloc[0]["RecipeId"] == 1
    assert diversified.iloc[1]["RecipeId"] == 3


def test_recommend_top_n_integrates_mmr_and_honors_requested_quantity():
    frame = pd.DataFrame({
        "RecipeId": [1, 2, 3, 4, 5, 6, 7, 8],
        "Name": ["apple banana", "apple banana sweet", "carrot soup", "rice beans", "fish lemon", "chocolate cake", "oat milk", "apple dessert"],
        "Description": ["apple banana"] * 8,
        "RecipeCategory": ["food"] * 8,
        "Keywords": [["food"]] * 8,
        "RecipeIngredientParts": [["apple", "banana"], ["apple", "banana"], ["carrot", "soup"], ["rice", "beans"], ["fish", "lemon"], ["chocolate", "cake"], ["oat", "milk"], ["apple", "dessert"]],
    })
    prepared = prepare_recipe_dataframe(frame)
    builder = build_tfidf_matrix(prepared)
    history = pd.DataFrame({"AuthorId": [5], "RecipeId": [8], "Rating": [5]})
    recommender = RecipeRecommender(
        prepared,
        history,
        builder,
        user_id=5,
        top_n=6,
        use_mmr=True,
        diversity_lambda=0.5,
    )
    results = recommender.recommend_top_n()
    assert len(results) == 6
    assert results["RecipeId"].is_unique
    assert results["explanation"].notna().all()
    assert 8 not in results["RecipeId"].tolist()


def test_experiment_runner_exports_results_without_test_profile_leakage(tmp_path):
    recipes = pd.DataFrame({
        "RecipeId": [1, 2, 3, 4, 5, 6],
        "Name": ["Tomato pasta", "Chicken salad", "Rice bowl", "Tomato soup", "Bean stew", "Fruit shake"],
        "Description": ["tomato pasta garlic", "chicken lettuce salad", "rice beans bowl", "tomato soup basil", "beans stew onion", "fruit banana shake"],
        "RecipeCategory": ["dinner", "lunch", "dinner", "dinner", "dinner", "drink"],
        "Keywords": [["pasta"], ["salad"], ["rice"], ["soup"], ["beans"], ["fruit"]],
        "RecipeIngredientParts": [["tomato", "pasta"], ["chicken", "lettuce"], ["rice", "beans"], ["tomato", "basil"], ["beans", "onion"], ["banana", "milk"]],
    })
    reviews = pd.DataFrame({
        "AuthorId": [10] * 4 + [20] * 4,
        "RecipeId": [1, 2, 3, 4, 2, 5, 6, 1],
        "Rating": [5, 4, 4, 5, 5, 4, 4, 5],
        "DateSubmitted": pd.to_datetime([
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
        ]),
    })
    recipe_path, review_path = tmp_path / "recipes.parquet", tmp_path / "reviews.parquet"
    recipes.to_parquet(recipe_path)
    reviews.to_parquet(review_path)
    output_dir = tmp_path / "results"

    summary, by_user, config = run_experiments(
        recipe_path=recipe_path,
        review_path=review_path,
        output_dir=output_dir,
        max_users=2,
    )

    assert set(summary["model"]) == {"RecipeMatch-TF-IDF", "Aleatório", "Popularidade"}
    assert by_user["n_train_reviews"].eq(3).all()
    assert by_user["eligible_candidates"].eq(3).all()
    assert by_user["n_test_relevant_candidates"].eq(1).all()
    assert config["chronological_split"] is True
    assert (output_dir / "metrics_summary.csv").exists()
    assert (output_dir / "metrics_by_user.csv").exists()
    assert json.loads((output_dir / "evaluation_config.json").read_text())["users_selected"] == 2
    assert json.loads((output_dir / "dataset_summary.json").read_text())["test_review_count"] == 2


def test_dataset_report_exports_real_dataframe_statistics(tmp_path, sample_recipes, sample_reviews):
    report = build_dataset_report(sample_recipes, sample_reviews, tmp_path)
    assert report["recipe_count"] == 4
    assert report["review_count"] == 5
    assert report["distinct_users"] == 2
    assert (tmp_path / "dataset_summary.json").exists()
    assert (tmp_path / "rating_distribution.csv").exists()
    assert (tmp_path / "dataset_missing_values.csv").exists()
    assert (tmp_path / "top_categories.csv").exists()


def test_qualitative_analysis_exports_requested_users_from_training_history(tmp_path):
    recipes = pd.DataFrame({
        "RecipeId": [1, 2, 3, 4, 5, 6, 7, 8],
        "Name": [f"Recipe {index}" for index in range(1, 9)],
        "Description": ["tomato pasta garlic", "chicken salad lettuce", "rice beans bowl", "tomato soup basil", "bean stew onion", "fruit banana shake", "oat milk breakfast", "fish lemon dinner"],
        "RecipeCategory": ["dinner"] * 8,
        "Keywords": [["easy"]] * 8,
        "RecipeIngredientParts": [["tomato", "pasta"], ["chicken", "lettuce"], ["rice", "beans"], ["tomato", "basil"], ["beans", "onion"], ["banana", "milk"], ["oat", "milk"], ["fish", "lemon"]],
    })
    reviews = pd.DataFrame({
        "AuthorId": [10] * 4 + [20] * 4,
        "RecipeId": [1, 2, 3, 4, 2, 5, 6, 1],
        "Rating": [5, 4, 3, 5, 5, 4, 3, 5],
        "DateSubmitted": pd.to_datetime([
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
        ]),
    })
    recipe_path, review_path = tmp_path / "recipes.parquet", tmp_path / "reviews.parquet"
    recipes.to_parquet(recipe_path)
    reviews.to_parquet(review_path)
    output = tmp_path / "qualitative.csv"

    result = run_qualitative_analysis(
        users=[10, 20],
        recipe_path=recipe_path,
        review_path=review_path,
        output_path=output,
    )

    assert set(result["user_id"]) == {10, 20}
    assert result["train_review_count"].eq(3).all()
    assert result["recipe_name"].notna().all()
    assert output.exists()

    automatic = run_qualitative_analysis(
        recipe_path=recipe_path,
        review_path=review_path,
        output_path=tmp_path / "automatic_qualitative.csv",
        random_state=7,
    )
    assert set(automatic["user_id"]) == {10, 20}
