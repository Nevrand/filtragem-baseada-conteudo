from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Permite executar `streamlit run main.py` com o diretório atual em `app/`.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.load_data import load_datasets
from src.data.preprocess import prepare_recipe_dataframe
from src.features.text_features import build_tfidf_matrix
from src.features.user_profile import build_user_profile
from src.recommendation.filters import filter_candidates
from src.recommendation.recommender import RecipeRecommender


@st.cache_data
def load_project_data():
    recipes, reviews = load_datasets()
    if recipes.empty or reviews.empty:
        return pd.DataFrame(), pd.DataFrame()
    recipes = prepare_recipe_dataframe(recipes)
    return recipes, reviews


st.title("RecipeMatch")
st.caption("Sistema de recomendação de receitas por conteúdo (Content-Based Filtering)")

recipes, reviews = load_project_data()
if recipes.empty or reviews.empty:
    st.warning("Dataset não encontrado ou incompleto. Coloque recipes.parquet e reviews.parquet em data/raw/.")
    st.stop()

user_ids = sorted(reviews["AuthorId"].dropna().unique().tolist())[:20]
selected_user = st.selectbox("Selecione um usuário", user_ids)

user_reviews = reviews[reviews["AuthorId"] == selected_user].sort_values("Rating", ascending=False)

st.subheader("Histórico do usuário")
st.write(f"Receitas avaliadas: {len(user_reviews)}")
st.write(user_reviews[["RecipeId", "Rating"]].head(10))

if "TotalTime" in recipes.columns:
    max_time = st.slider("Tempo máximo de preparo (minutos)", 15, 300, 60, 15)
else:
    max_time = None

avoid_input = st.text_input("Ingredientes a evitar (separados por vírgula)", "")
available_input = st.text_input("Ingredientes disponíveis em casa (separados por vírgula)", "")
result_count = st.slider("Número de recomendações", 3, 10, 5)

if st.button("Gerar recomendações"):
    tfidf = build_tfidf_matrix(recipes)
    recommender = RecipeRecommender(recipes, reviews, tfidf, user_id=selected_user, top_n=result_count)
    user_vector, _ = build_user_profile(user_reviews, recipes, tfidf)
    if user_vector is None:
        st.info("O usuário não possui histórico suficiente para construir um perfil de preferência.")
    else:
        candidates = filter_candidates(
            recipes,
            user_reviews,
            max_minutes=max_time,
            avoided_ingredients=[x.strip() for x in avoid_input.split(",") if x.strip()],
            available_ingredients=[x.strip() for x in available_input.split(",") if x.strip()],
            ingredient_column="RecipeIngredientParts",
            recipe_id_column="RecipeId",
        )
        ranked = recommender.compute_scores(user_vector, candidate_df=candidates)
        for _, row in ranked.head(result_count).iterrows():
            st.markdown(f"### {row['RecipeId']} - {row.get('Name', 'Receita')}")
            st.write(row.get("explanation", ""))
            st.write("Score:", round(float(row.get("score", 0.0)), 4))
