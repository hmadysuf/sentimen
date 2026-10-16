"""Modul prediksi komentar baru & visualisasi."""
import os
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .preprocess import clean_text, normalize_text, remove_stopwords, stem_text
from .lexicon import score_text, label_from_score


def predict_text(text: str, model_dir: str):
    """Prediksi satu komentar dengan NB & SVM."""
    nb = joblib.load(os.path.join(model_dir, "naive_bayes.pkl"))
    svm = joblib.load(os.path.join(model_dir, "svm.pkl"))
    tfidf = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))

    # Preprocessing
    cleaned = clean_text(text)
    normalized = normalize_text(cleaned)
    no_stop = remove_stopwords(normalized)
    stemmed = stem_text(no_stop)
    tokens = stemmed.split()

    vec = tfidf.transform([stemmed])
    pred_nb = nb.predict(vec)[0]
    pred_svm = svm.predict(vec)[0]

    # Lexicon-based
    lex_score = score_text(tokens)
    lex_label = label_from_score(lex_score)

    return {
        "text": text,
        "cleaned": cleaned,
        "stemmed": stemmed,
        "naive_bayes": pred_nb,
        "svm": pred_svm,
        "lexicon": lex_label,
        "lexicon_score": lex_score,
    }


def plot_sentiment_distribution(df: pd.DataFrame, label_col: str, out_path: str):
    plt.figure(figsize=(7, 4))
    order = ["positif", "netral", "negatif"]
    counts = df[label_col].value_counts().reindex(order).fillna(0)
    colors = ["#2ecc71", "#95a5a6", "#e74c3c"]
    sns.barplot(x=counts.index, y=counts.values, palette=colors)
    plt.title("Distribusi Sentimen Komentar YouTube tentang MBG")
    plt.xlabel("Kelas Sentimen")
    plt.ylabel("Jumlah Komentar")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 0.5, str(int(v)), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_confusion_matrix(cm, labels, title, out_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_comment_vs_reply(df: pd.DataFrame, static_dir: str):
    """Grafik perbandingan sentimen komentar top-level vs reply."""
    if "type" not in df.columns:
        return
    plt.figure(figsize=(8, 4))
    order = ["positif", "netral", "negatif"]
    pivot = df.groupby(["type", "label"]).size().unstack(fill_value=0).reindex(
        columns=order, fill_value=0
    )
    pivot.plot(kind="bar", color=["#2ecc71", "#95a5a6", "#e74c3c"], ax=plt.gca())
    plt.title("Sentimen Komentar Top-Level vs Reply")
    plt.xlabel("Tipe")
    plt.ylabel("Jumlah")
    plt.xticks(rotation=0)
    plt.legend(title="Sentimen")
    plt.tight_layout()
    out = os.path.join(static_dir, "charts", "comment_vs_reply.png")
    plt.savefig(out, dpi=120)
    plt.close()