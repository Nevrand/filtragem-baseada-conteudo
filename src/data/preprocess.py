from __future__ import annotations

import re
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

DEFAULT_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "of", "to", "in", "on",
    "with", "by", "as", "at", "from", "it", "is", "be", "are", "was", "were", "this", "that",
    "these", "those", "into", "out", "over", "under", "about", "after", "before", "during", "not",
    "no", "yes", "you", "your", "we", "our", "they", "them", "their", "his", "her", "she", "he",
    "i", "me", "my", "mine", "do", "does", "did", "have", "has", "had", "will", "would", "could",
    "should", "may", "might", "can", "make", "made", "using", "use", "recipe", "recipes", "ingredients",
    "ingredient", "dish", "food", "fresh", "cup", "cups", "tablespoon", "tablespoons", "teaspoon",
    "teaspoons", "minute", "minutes", "hour", "hours", "day", "days", "mix", "stir", "bake", "cook",
    "cooked", "serving", "servings"
}


def normalize_text(value) -> str:
    """Normaliza texto em minúsculas, remove pontuação excessiva e tokens vazios."""
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        value = " ".join(normalize_text(item) for item in value if not _is_missing_scalar(item))
    elif _is_missing_scalar(value):
        return ""
    text = str(value).lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_missing_scalar(value) -> bool:
    """Verifica ausência sem avaliar arrays como booleanos."""
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def clean_text_series(series: pd.Series, remove_stopwords: bool = True) -> pd.Series:
    """Aplica limpeza textual em uma série inteira."""
    cleaned = series.apply(normalize_text)
    if remove_stopwords:
        cleaned = cleaned.apply(remove_stopwords_from_text)
    return cleaned


def remove_stopwords_from_text(text: str, stopwords: Iterable[str] | None = None) -> str:
    """Remove stopwords em português e inglês, mantendo termos úteis para recomendação."""
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    stop = set(stopwords) if stopwords is not None else DEFAULT_STOPWORDS
    filtered = [token for token in tokens if token not in stop and len(token) > 1]
    return " ".join(filtered)


def coerce_ingredient_list(value) -> List[str]:
    """Converte listas, strings e estruturas semelhantes para lista de ingredientes."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        items = value.tolist()
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]

    cleaned = []
    for item in items:
        if item is None or (isinstance(item, float) and pd.isna(item)):
            continue
        token = str(item).strip()
        if token:
            cleaned.append(normalize_text(token))
    return cleaned


def parse_duration_to_minutes(value) -> float | None:
    """Converte durações ISO compatíveis ou texto simples em minutos.

    Durações ausentes/inválidas retornam ``None``. Segundos são preservados como
    frações de minuto e durações ISO com dias (por exemplo, ``P1DT2H``) são aceitas.
    Unidades ISO não suportadas (como anos, meses e semanas) não são inferidas como
    minutos, pois não têm conversão fixa para minutos sem contexto adicional.
    """
    if value is None or _is_missing_scalar(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().upper()
    if not text:
        return None

    iso_match = re.fullmatch(
        r"P(?:(?P<days>\d+(?:\.\d+)?)D)?"
        r"(?:T(?:(?P<hours>\d+(?:\.\d+)?)H)?"
        r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?"
        r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?",
        text,
    )
    if iso_match:
        parts = iso_match.groupdict()
        if not any(parts.values()):
            return None
        return (
            float(parts["days"] or 0) * 24 * 60
            + float(parts["hours"] or 0) * 60
            + float(parts["minutes"] or 0)
            + float(parts["seconds"] or 0) / 60
        )

    if text.startswith("P"):
        return None

    if re.search(r"\d", text):
        values = re.findall(r"(\d+(?:\.\d+)?)", text)
        if not values:
            return None
        if re.search(r"\b(?:HOUR|HOURS|HR|HRS)\b", text):
            return float(values[0]) * 60 + (float(values[1]) if len(values) > 1 else 0)
        if re.search(r"\b(?:DAY|DAYS)\b", text):
            return float(values[0]) * 24 * 60
        if re.search(r"\b(?:SECOND|SECONDS|SEC|SECS)\b", text):
            return float(values[0]) / 60
        if re.search(r"\b(?:MINUTE|MINUTES|MIN|MINS)\b", text):
            return float(values[0])
        return float(values[0])

    return None


def build_recipe_text(row: pd.Series, fields: Sequence[str] | None = None) -> str:
    """Combina campos textuais relevantes para a representação TF-IDF da receita."""
    selected = fields or ["Name", "Description", "RecipeCategory", "Keywords", "RecipeIngredientParts"]
    text_parts = []
    for field in selected:
        if field not in row.index:
            continue
        value = row[field]
        if field == "Keywords" and isinstance(value, (list, tuple, set)):
            value = " ".join(str(item) for item in value)
        if field == "RecipeIngredientParts" and isinstance(value, (list, tuple, set)):
            value = " ".join(str(item) for item in value)
        if field == "RecipeIngredientParts" and isinstance(value, str):
            value = value.replace("[", " ").replace("]", " ")
        text_parts.append(normalize_text(value))
    return " ".join(part for part in text_parts if part)


def prepare_recipe_dataframe(
    recipe_df: pd.DataFrame,
    text_fields: Sequence[str] | None = None,
    ingredient_column: str = "RecipeIngredientParts",
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Aplica limpeza, normalização e simples pré-processamento aos dados das receitas."""
    if recipe_df.empty:
        return recipe_df.copy()

    output = recipe_df.copy()
    for column in ["Name", "Description", "RecipeCategory", "RecipeIngredientParts", "Keywords"]:
        if column in output.columns:
            output[column] = output[column].fillna("")

    if ingredient_column in output.columns:
        output[ingredient_column] = output[ingredient_column].apply(lambda value: [normalize_text(item) for item in coerce_ingredient_list(value)])

    if drop_duplicates:
        output = output.drop_duplicates(subset=["RecipeId"], keep="first") if "RecipeId" in output.columns else output.drop_duplicates()

    text_fields = list(text_fields or ["Name", "Description", "RecipeCategory", "Keywords", "RecipeIngredientParts"])
    output["combined_text"] = output.apply(lambda row: build_recipe_text(row, text_fields), axis=1)
    output["combined_text"] = output["combined_text"].apply(remove_stopwords_from_text)
    return output


__all__ = [
    "DEFAULT_STOPWORDS",
    "normalize_text",
    "clean_text_series",
    "remove_stopwords_from_text",
    "coerce_ingredient_list",
    "parse_duration_to_minutes",
    "build_recipe_text",
    "prepare_recipe_dataframe",
]
