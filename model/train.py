"""Тренування моделі + збереження разом зі статистикою фіч (для drift-детектора).

Запуск: python model/train.py  → model/model.pkl
Використовується і при білді образу, і в retrain-пайплайні (GitLab CI).
"""
import os
import pickle

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

OUT = os.path.join(os.path.dirname(__file__), "model.pkl")


def main():
    data = load_iris()
    X, y = data.data, data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=500)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))

    # Статистика тренувальних фіч — еталон для виявлення дрейфу
    bundle = {
        "model": model,
        "feature_means": X.mean(axis=0).tolist(),
        "feature_stds": X.std(axis=0).tolist(),
        "target_names": data.target_names.tolist(),
        "accuracy": float(acc),
    }
    with open(OUT, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Trained LogisticRegression, accuracy={acc:.4f} -> {OUT}")


if __name__ == "__main__":
    main()
