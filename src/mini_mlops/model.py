from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


def build_model(seed: int = 42) -> Pipeline:
    """Build the complete text classification pipeline."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def train_model(
    model: Pipeline,
    train_df: pd.DataFrame,
) -> Pipeline:
    """Fit the model using training data."""

    model.fit(
        train_df["text"],
        train_df["label"],
    )

    return model


def evaluate_model(
    model: Pipeline,
    test_df: pd.DataFrame,
) -> dict[str, float]:
    """Evaluate the model on held-out test data."""

    predictions = model.predict(test_df["text"])

    return {
        "accuracy": accuracy_score(
            test_df["label"],
            predictions,
        ),
        "macro_f1": f1_score(
            test_df["label"],
            predictions,
            average="macro",
        ),
    }