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

# ── CSS mínimo — apenas elementos pontuais, compatível com dark/light mode ───
st.markdown(
    """
    <style>
    /* Badge de ranking: fundo fixo colorido com texto branco — legível em qualquer tema */
    .rm-rank-badge {
        display: inline-block;
        background: #e8603c;
        color: #fff;
        font-size: 0.74rem;
        font-weight: 700;
        border-radius: 20px;
        padding: 2px 10px;
        margin-right: 6px;
        vertical-align: middle;
    }

    /* Pills de ingrediente: fundo semi-transparente, herda a cor do tema */
    .rm-pill {
        display: inline-block;
        background: rgba(128, 128, 128, 0.15);
        font-size: 0.74rem;
        border-radius: 20px;
        padding: 2px 9px;
        margin: 2px;
    }

    /* Labels de seção na sidebar */
    .rm-sidebar-section {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.5;
        margin: 14px 0 4px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cabeçalho (nativo — respeita o tema dark/light automaticamente) ──────────
st.title("RecipeMatch")
st.caption("Recomendações personalizadas por conteúdo, com TF-IDF e similaridade de cosseno.")
st.divider()


# ── Funções cacheadas (idênticas ao main.py) ─────────────────────────────────
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


# ── Verificação do dataset ────────────────────────────────────────────────────
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

# ── Seleção de usuário ────────────────────────────────────────────────────────
review_counts = reviews.groupby("AuthorId").size()
minimum_user_history = 50
user_options = review_counts[review_counts >= minimum_user_history].sort_values(ascending=False).index.tolist()
if not user_options:
    st.error(f"Não há usuários com pelo menos {minimum_user_history} avaliações para a demonstração.")
    st.stop()

selected_user = st.selectbox(
    "Usuário",
    user_options,
    format_func=lambda user_id: f"{user_id} ({int(review_counts.loc[user_id])} avaliações)",
    help=(
        f"Exibe apenas usuários com pelo menos {minimum_user_history} avaliações, "
        "garantindo histórico suficiente para a recomendação."
    ),
)

# ── Histórico do usuário ──────────────────────────────────────────────────────
user_reviews = reviews[reviews["AuthorId"] == selected_user].copy()
user_mean = float(user_reviews["Rating"].mean())

st.markdown("### Histórico")

# st.metric respeita o tema dark/light automaticamente
col_a, col_b, _ = st.columns([1, 1, 4])
with col_a:
    st.metric("Avaliações", len(user_reviews))
with col_b:
    st.metric("Nota média", f"{user_mean:.2f}")

with st.expander("Ver histórico completo", expanded=False):
    history_columns = ["RecipeId", "Rating"]
    if "DateSubmitted" in user_reviews.columns:
        history_columns.append("DateSubmitted")
    history = user_reviews[history_columns].merge(
        recipes[["RecipeId", "Name", "RecipeIngredientParts"]],
        on="RecipeId",
        how="left",
    )
    history["Ingredients"] = history["RecipeIngredientParts"].map(
        lambda value: ", ".join(coerce_ingredient_list(value))
    )
    history = history.rename(columns={"Name": "Recipe", "Rating": "Rating"})
    display_columns = ["Recipe", "Rating", "Ingredients"]
    if "DateSubmitted" in history.columns:
        display_columns.append("DateSubmitted")
    st.dataframe(
        history[display_columns].sort_values("Rating", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ── Sidebar: preferências e filtros ──────────────────────────────────────────
with st.sidebar:
    st.header("Preferências e filtros")

    st.markdown('<p class="rm-sidebar-section">Ingredientes</p>', unsafe_allow_html=True)
    avoid_input = st.text_input(
        "Ingredientes a evitar (separados por vírgula)",
        placeholder="ex: tomato, garlic, onion",
        help="Receitas com esses ingredientes serão removidas dos resultados.",
    )
    available_input = st.text_input(
        "Ingredientes disponíveis (separados por vírgula)",
        placeholder="ex: chicken, rice, butter",
        help="Mantém receitas que contenham ao menos um deles.",
    )

    st.markdown('<p class="rm-sidebar-section">Tempo e quantidade</p>', unsafe_allow_html=True)
    max_time = st.number_input(
        "Tempo máximo (minutos)",
        min_value=1,
        max_value=10080,
        value=60,
        help="Usa TotalTime; receitas sem tempo verificável são excluídas quando este filtro está ativo.",
    )
    result_count = st.slider(
        "Quantidade de recomendações",
        min_value=1,
        max_value=20,
        value=5,
    )

    st.markdown('<p class="rm-sidebar-section">Diversidade</p>', unsafe_allow_html=True)
    use_mmr = st.checkbox("Promover diversidade (MMR)", value=False)
    diversity_lambda = st.slider(
        "Equilíbrio relevância/diversidade (λ)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        disabled=not use_mmr,
        help="Valores próximos de 1 priorizam relevância; valores próximos de 0 priorizam diversidade.",
    )

    st.divider()
    gerar = st.button("🔍 Gerar recomendações", type="primary", use_container_width=True)

# ── Geração das recomendações ─────────────────────────────────────────────────
if gerar:
    # Captura os valores dos widgets no momento do clique
    _result_count = result_count
    _max_time = max_time
    _avoid = [item.strip() for item in avoid_input.split(",") if item.strip()]
    _available = [item.strip() for item in available_input.split(",") if item.strip()]
    _use_mmr = use_mmr
    _lambda = diversity_lambda

    with st.spinner("Calculando recomendações..."):
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
            top_n=_result_count,
            diversity_lambda=_lambda,
            use_mmr=_use_mmr,
        )
        candidates = filter_candidates(
            recipes,
            user_reviews,
            max_minutes=_max_time,
            avoided_ingredients=_avoid,
            available_ingredients=_available,
        )
        ranked = recommender.compute_scores(
            profile,
            candidates,
            top_n=None if _use_mmr else _result_count,
            explain=not _use_mmr,
        )
        if ranked.empty:
            st.warning("Nenhuma receita atende aos filtros. Ajuste os critérios; nenhum filtro foi removido.")
            st.stop()
        if _use_mmr:
            ranked = recommender.mmr_diversify(ranked, _lambda)
            ranked = ranked.head(_result_count).copy()
            ranked = recommender.explain_recommendations(ranked)

    # ── Exibição dos resultados ───────────────────────────────────────────────
    st.markdown("### Recomendações")

    for rank_i, row in enumerate(ranked.head(_result_count).itertuples(index=False), start=1):
        # Imagem (mesma lógica do main.py)
        image_values = row.Images if hasattr(row, "Images") else None
        image_url = None
        if isinstance(image_values, np.ndarray):
            image_values = image_values.tolist()
        if isinstance(image_values, (list, tuple)):
            image_url = next(
                (str(v) for v in image_values if str(v).startswith(("https://", "http://"))),
                None,
            )

        # Metadados
        ingredients = coerce_ingredient_list(getattr(row, "RecipeIngredientParts", []))
        minutes = parse_duration_to_minutes(getattr(row, "TotalTime", None))
        category = getattr(row, "RecipeCategory", "Não informada") or "Não informada"
        time_str = f"{minutes:g} min" if minutes is not None else "Não informado"

        # Pills de ingredientes (máx 8)
        pills = "".join(
            f'<span class="rm-pill">{ing}</span>' for ing in ingredients[:8]
        )
        if len(ingredients) > 8:
            pills += f'<span class="rm-pill">+{len(ingredients) - 8} mais</span>'

        # Card nativo (st.container respeita o tema dark/light)
        with st.container(border=True):
            if image_url:
                col_text, col_img = st.columns([3, 1])
                with col_img:
                    st.image(image_url, width=320)
                with col_text:
                    st.markdown(
                        f'<span class="rm-rank-badge">#{rank_i}</span> **{row.Name}**',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"{category}  ·  ⏱ {time_str}  ·  Compatibilidade: {row.score:.4f}")
                    if getattr(row, "Description", None):
                        st.write(row.Description)
                    st.markdown(f"**Ingredientes:** {pills}", unsafe_allow_html=True)
                    st.write(row.explanation)
            else:
                st.markdown(
                    f'<span class="rm-rank-badge">#{rank_i}</span> **{row.Name}**',
                    unsafe_allow_html=True,
                )
                st.caption(f"{category}  ·  ⏱ {time_str}  ·  Compatibilidade: {row.score:.4f}")
                if getattr(row, "Description", None):
                    st.write(row.Description)
                st.markdown(f"**Ingredientes:** {pills}", unsafe_allow_html=True)
                st.write(row.explanation)
