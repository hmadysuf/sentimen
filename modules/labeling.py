"""Pelabelan otomatis berbasis leksikon."""
import pandas as pd
from .lexicon import score_text, label_from_score


def label_dataframe(df: pd.DataFrame, tokens_col: str = "tokens") -> pd.DataFrame:
    df = df.copy()
    df["score"] = df[tokens_col].apply(score_text)
    df["label"] = df["score"].apply(label_from_score)
    return df