from __future__ import annotations

from typing import Tuple

import pandas as pd


def split_reviews_by_user(
    reviews_df: pd.DataFrame,
    test_fraction: float = 0.2,
    random_state: int = 42,
    user_column: str = "AuthorId",
    rating_column: str = "Rating",
    date_column: str | None = "DateSubmitted",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide avaliações em treino e teste por usuário.

    Se há data confiável, usa as avaliações mais recentes como teste; caso contrário,
    faz uma divisão aleatória reprodutível por usuário. Usuários com histórico muito
    pequeno permanecem no conjunto de treino para que o perfil seja construído.
    """
    if reviews_df.empty:
        return reviews_df.copy(), reviews_df.copy()

    working = reviews_df.copy()
    if user_column not in working.columns:
        raise ValueError(f"Coluna '{user_column}' não encontrada para divisão por usuário.")

    if date_column is not None and date_column in working.columns:
        working = working.sort_values(date_column, kind="mergesort").copy()
        train_frames = []
        test_frames = []
        for _, group in working.groupby(user_column, sort=False):
            if len(group) <= 1:
                train_frames.append(group)
                continue
            split_index = max(1, int(len(group) * (1.0 - test_fraction)))
            train_frames.append(group.iloc[:split_index])
            test_frames.append(group.iloc[split_index:])
        return pd.concat(train_frames, ignore_index=True), pd.concat(test_frames, ignore_index=True)

    train_frames = []
    test_frames = []
    for _, group in working.groupby(user_column, sort=False):
        if len(group) <= 1:
            train_frames.append(group)
            continue
        shuffled = group.sample(frac=1, random_state=random_state + int(group[user_column].iloc[0]))
        split_index = max(1, int(len(shuffled) * (1.0 - test_fraction)))
        train_frames.append(shuffled.iloc[:split_index])
        test_frames.append(shuffled.iloc[split_index:])
    return pd.concat(train_frames, ignore_index=True), pd.concat(test_frames, ignore_index=True)


__all__ = ["split_reviews_by_user"]
