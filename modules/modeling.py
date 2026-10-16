"""Modul training Naive Bayes dan SVM."""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score,
    confusion_matrix, classification_report
)


def train_models(df: pd.DataFrame, text_col: str, label_col: str,
                 model_dir: str, metrics_path: str):
    """Melatih Naive Bayes & SVM, simpan model + metrik evaluasi."""
    X = df[text_col].astype(str).values
    y = df[label_col].astype(str).values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    results = {}

    # === Naive Bayes ===
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_train_tfidf, y_train)
    y_pred_nb = nb.predict(X_test_tfidf)

    # === SVM (LinearSVC) ===
    svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
    svm.fit(X_train_tfidf, y_train)
    y_pred_svm = svm.predict(X_test_tfidf)

    labels = sorted(set(y))

    for name, y_pred in [("naive_bayes", y_pred_nb), ("svm", y_pred_svm)]:
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=labels, average="macro", zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
        results[name] = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision_macro": round(p, 4),
            "recall_macro": round(r, 4),
            "f1_macro": round(f1, 4),
            "confusion_matrix": cm,
            "labels": labels,
            "classification_report": classification_report(
                y_test, y_pred, labels=labels, zero_division=0, output_dict=True
            ),
        }

    results["dataset_info"] = {
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "label_distribution": df[label_col].value_counts().to_dict(),
    }

    # Simpan model
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(nb, os.path.join(model_dir, "naive_bayes.pkl"))
    joblib.dump(svm, os.path.join(model_dir, "svm.pkl"))
    joblib.dump(tfidf, os.path.join(model_dir, "tfidf_vectorizer.pkl"))

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results