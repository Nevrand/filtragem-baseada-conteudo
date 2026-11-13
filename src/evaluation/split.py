from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd


def split_reviews_by_user(
    reviews_df: pd.DataFrame,
    test_fraction: float = 0.2,
    random_state: int = 42,
    user_column: str = "AuthorId",
    rating_column: str = "Rating",
    date_column: str | None = "DateSubmitted",
    min_train_reviews: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa pares usuário-receita, priorizando holdout cronológico.

    A divisão cronológica só é usada quando cada par tem pelo menos uma data válida.
    Avaliações repetidas de um par são reduzidas a uma única interação: usa-se a
    média das notas válidas de 1 a 5; se não houver nota válida, preserva-se 0 caso
    exista uma interação sem nota por estrelas. A linha com data mais recente
    representa os demais campos do par. Usuários sem pares suficientes permanecem
    apenas no treino.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction deve estar entre 0 e 1.")
    if min_train_reviews < 1:
        raise ValueError("min_train_reviews deve ser pelo menos 1.")
    if reviews_df.empty:
        return reviews_df.copy(), reviews_df.copy()
    if user_column not in reviews_df.columns or rating_column not in reviews_df.columns:
        raise ValueError(f"Colunas obrigatórias ausentes: {user_column!r} e/ou {rating_column!r}.")

    working = reviews_df.copy()
    has_date_column = date_column is not None and date_column in working.columns
    if has_date_column:
        working["_split_date"] = pd.to_datetime(
            working[date_column], errors="coerce", utc=True, format="mixed"
        )

    pair_columns = [user_column, "RecipeId"]
    if "RecipeId" not in working.columns:
        raise ValueError("A coluna obrigatória 'RecipeId' está ausente.")
    working["_input_order"] = range(len(working))
    working["_numeric_rating"] = pd.to_numeric(working[rating_column], errors="coerce")
    consolidated_rows = []
    for _, pair in working.groupby(pair_columns, sort=False, dropna=False):
        if has_date_column and pair["_split_date"].notna().any():
            representative = pair.sort_values(
                "_split_date",
                kind="mergesort",
                na_position="first",
            ).iloc[-1].copy()
        else:
            representative = pair.sort_values("_input_order", kind="mergesort").iloc[-1].copy()
        valid_ratings = pair["_numeric_rating"].where(pair["_numeric_rating"].between(1, 5)).dropna()
        if not valid_ratings.empty:
            representative[rating_column] = float(valid_ratings.mean())
        elif pair["_numeric_rating"].eq(0).any():
            representative[rating_column] = 0
        else:
            representative[rating_column] = np.nan
        consolidated_rows.append(representative)
    working = pd.DataFrame(consolidated_rows).reset_index(drop=True)
    use_dates = has_date_column and bool(working["_split_date"].notna().all())

    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    for group_index, (_, group) in enumerate(working.groupby(user_column, sort=True, dropna=False)):
        if len(group) < min_train_reviews + 1:
            train_frames.append(group)
            continue

        if use_dates:
            group = group.sort_values("_split_date", kind="mergesort")
            test_count = max(1, ceil(len(group) * test_fraction))
            split_at = len(group) - min(test_count, len(group) - min_train_reviews)
            cutoff = group["_split_date"].iloc[split_at]
            earlier = group["_split_date"] < cutoff
            if int(earlier.sum()) < min_train_reviews:
                train_frames.append(group)
                continue
            train_frames.append(group.loc[earlier])
            test_frames.append(group.loc[~earlier])
        else:
            group = group.sample(frac=1, random_state=random_state + group_index)
            test_count = max(1, ceil(len(group) * test_fraction))
            test_count = min(test_count, len(group) - min_train_reviews)
            split_at = len(group) - test_count
            train_frames.append(group.iloc[:split_at])
            test_frames.append(group.iloc[split_at:])

    train = pd.concat(train_frames, ignore_index=True) if train_frames else working.iloc[:0].copy()
    test = pd.concat(test_frames, ignore_index=True) if test_frames else working.iloc[:0].copy()
    temporary_columns = ["_split_date", "_input_order", "_numeric_rating"]
    return train.drop(columns=temporary_columns, errors="ignore"), test.drop(columns=temporary_columns, errors="ignore")


__all__ = ["split_reviews_by_user"]
