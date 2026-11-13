from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RecipeMatch", page_icon="🍲", layout="wide")

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.load_data import DEFAULT_RECIPE_PATH, DEFAULT_REVIEW_PATH, load_datasets
from src.data.preprocess import coerce_ingredient_list, parse_duration_to_minutes, prepare_recipe_dataframe
from src.features.text_features import build_tfidf_matrix
from src.features.user_profile import build_user_profile
from src.recommendation.filters import filter_candidates
from src.recommendation.recommender import RecipeRecommender


@st.cache_data(show_spinner="Carregando e preparando o dataset...")
def load_project_data(recipe_path: str, review_path: str, recipe_mtime: int, review_mtime: int):
    del recipe_mtime, review_mtime
    recipes, reviews = load_datasets(
        recipe_path=recipe_path,
        review_path=review_path,
        recipe_required=(
            "RecipeId",
            "Name",
            "Description",
            "RecipeCategory",
            "Keywords",
            "RecipeIngredientParts",
            "TotalTime",
        ),
        review_required=("RecipeId", "AuthorId", "Rating"),
        recipe_columns=(
            "RecipeId",
            "Name",
            "Description",
            "RecipeCategory",
            "Keywords",
            "RecipeIngredientParts",
            "PrepTime",
            "CookTime",
            "TotalTime",
            "Images",
        ),
        review_columns=("RecipeId", "AuthorId", "Rating", "DateSubmitted"),
    )
    if recipes.empty or reviews.empty:
        return pd.DataFrame(), pd.DataFrame()
    return prepare_recipe_dataframe(recipes), reviews


@st.cache_resource(show_spinner="Construindo modelo TF-IDF (uma vez por versão dos dados)...")
def load_text_model(
    recipe_path: str,
    review_path: str,
    recipe_mtime: int,
    review_mtime: int,
    recipe_size: int,
    review_size: int,
):
    del recipe_size, review_size
    recipes, _ = load_project_data(
        recipe_path,
        review_path,
        recipe_mtime,
        review_mtime,
    )
    return build_tfidf_matrix(recipes, max_features=5000)


st.title("RecipeMatch")
st.caption("Recomendações personalizadas por conteúdo, com TF-IDF e similaridade de cosseno.")

if not DEFAULT_RECIPE_PATH.exists() or not DEFAULT_REVIEW_PATH.exists():
    st.error(
        "Dataset não encontrado. Coloque `recipes.parquet` e `reviews.parquet` em `data/raw/`."
    )
    st.stop()

recipes, reviews = load_project_data(
    str(DEFAULT_RECIPE_PATH),
    str(DEFAULT_REVIEW_PATH),
    DEFAULT_RECIPE_PATH.stat().st_mtime_ns,
    DEFAULT_REVIEW_PATH.stat().st_mtime_ns,
)
if recipes.empty or reviews.empty:
    st.error("Os arquivos de dados foram encontrados, mas não contêm receitas e avaliações utilizáveis.")
    st.stop()

review_counts = reviews.groupby("AuthorId").size()
numeric_ratings = pd.to_numeric(reviews["Rating"], errors="coerce")
valid_rating_counts = (
    reviews.assign(_valid_rating=numeric_ratings.between(1, 5))
    .groupby("AuthorId")["_valid_rating"]
    .sum()
)
minimum_user_history = 50
user_options = review_counts[review_counts >= minimum_user_history].sort_values(ascending=False).index.tolist()
if not user_options:
    st.error(f"Não há usuários com pelo menos {minimum_user_history} avaliações para a demonstração.")
    st.stop()

st.caption(
    f"O seletor inclui todos os usuários com pelo menos {minimum_user_history} interações; "
    "esse critério reduz opções com histórico insuficiente sem limitar a lista aos primeiros IDs."
)
selected_user = st.selectbox(
    "Usuário",
    user_options,
    format_func=lambda user_id: (
        f"{user_id} ({int(review_counts.loc[user_id])} interações; "
        f"{int(valid_rating_counts.loc[user_id])} avaliações por estrelas)"
    ),
)
user_reviews = reviews[reviews["AuthorId"] == selected_user].copy()
user_ratings = pd.to_numeric(user_reviews["Rating"], errors="coerce")
valid_user_ratings = user_ratings[user_ratings.between(1, 5)]
user_mean = float(valid_user_ratings.mean()) if not valid_user_ratings.empty else None
st.subheader("Histórico")
average_text = f"{user_mean:.2f}" if user_mean is not None else "sem avaliações válidas"
st.write(
    f"**{len(user_reviews)} interações** · "
    f"**{len(valid_user_ratings)} avaliações por estrelas** · "
    f"média **{average_text}**"
)

history_columns = ["RecipeId", "Rating"]
if "DateSubmitted" in user_reviews.columns:
    history_columns.append("DateSubmitted")
history = user_reviews[history_columns].merge(
    recipes[["RecipeId", "Name", "RecipeIngredientParts"]],
    on="RecipeId",
    how="left",
)
history["_rating_sort"] = pd.to_numeric(history["Rating"], errors="coerce")
history["Rating"] = pd.to_numeric(history["Rating"], errors="coerce").map(
    lambda rating: (
        "Sem avaliação por estrelas"
        if rating == 0
        else rating
        if pd.notna(rating) and 1 <= rating <= 5
        else "Nota inválida"
    )
)
history["Ingredients"] = history["RecipeIngredientParts"].map(
    lambda value: ", ".join(coerce_ingredient_list(value))
)
history = history.rename(columns={"Name": "Recipe", "Rating": "Rating"})
display_columns = ["Recipe", "Rating", "Ingredients"]
if "DateSubmitted" in history.columns:
    display_columns.append("DateSubmitted")
st.dataframe(
    history.sort_values("_rating_sort", ascending=False, na_position="last")[display_columns],
    use_container_width=True,
)

with st.sidebar:
    st.header("Preferências e filtros")
    avoid_input = st.text_input("Ingredientes a evitar (separados por vírgula)")
    available_input = st.text_input("Ingredientes disponíveis (separados por vírgula)")
    max_time = st.number_input("Tempo máximo (minutos)", min_value=1, max_value=10080, value=60)
    result_count = st.slider("Quantidade de recomendações", min_value=1, max_value=20, value=5)
    use_mmr = st.checkbox("Promover diversidade (MMR)", value=False)
    diversity_lambda = st.slider(
        "Equilíbrio relevância/diversidade (λ)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        disabled=not use_mmr,
    )

if st.button("Gerar recomendações", type="primary"):
    tfidf = load_text_model(
        str(DEFAULT_RECIPE_PATH),
        str(DEFAULT_REVIEW_PATH),
        DEFAULT_RECIPE_PATH.stat().st_mtime_ns,
        DEFAULT_REVIEW_PATH.stat().st_mtime_ns,
        DEFAULT_RECIPE_PATH.stat().st_size,
        DEFAULT_REVIEW_PATH.stat().st_size,
    )
    profile, _ = build_user_profile(user_reviews, recipes, tfidf)
    if profile is None:
        st.info("O histórico não contém avaliações informativas suficientes para construir um perfil.")
        st.stop()

    recommender = RecipeRecommender(
        recipes_df=recipes,
        reviews_df=user_reviews,
        tfidf_builder=tfidf,
        user_id=selected_user,
        top_n=result_count,
        diversity_lambda=diversity_lambda,
        use_mmr=use_mmr,
    )
    candidates = filter_candidates(
        recipes,
        user_reviews,
        max_minutes=max_time,
        avoided_ingredients=[item.strip() for item in avoid_input.split(",") if item.strip()],
        available_ingredients=[item.strip() for item in available_input.split(",") if item.strip()],
    )
    ranked = recommender.compute_scores(
        profile,
        candidates,
        top_n=None if use_mmr else result_count,
        explain=not use_mmr,
    )
    if ranked.empty:
        st.warning("Nenhuma receita atende aos filtros. Ajuste os critérios; nenhum filtro foi removido.")
        st.stop()
    if use_mmr:
        ranked = recommender.mmr_diversify(ranked, diversity_lambda)
        ranked = ranked.head(result_count).copy()
        ranked = recommender.explain_recommendations(ranked)
    st.subheader("Recomendações")
    for row in ranked.head(result_count).itertuples(index=False):
        st.markdown(f"### {row.Name}")
        image_values = row.Images if hasattr(row, "Images") else None
        image_url = None
        if isinstance(image_values, np.ndarray):
            image_values = image_values.tolist()
        if isinstance(image_values, (list, tuple)):
            image_url = next(
                (str(value) for value in image_values if str(value).startswith(("https://", "http://"))),
                None,
            )
        if image_url:
            st.image(image_url, width=320)
        if getattr(row, "Description", None):
            st.write(row.Description)
        st.write("**Categoria:**", getattr(row, "RecipeCategory", "Não informada"))
        ingredients = coerce_ingredient_list(getattr(row, "RecipeIngredientParts", []))
        st.write("**Ingredientes:**", ", ".join(ingredients) or "Não informados")
        minutes = parse_duration_to_minutes(getattr(row, "TotalTime", None))
        st.write("**Tempo total:**", f"{minutes:g} min" if minutes is not None else "Não informado")
        st.write(f"**Compatibilidade:** {row.score:.4f}")
        st.write(row.explanation)
        st.divider()
