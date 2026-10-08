
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from mini_mlops.model import build_model


def test_build_model_structure():
    model = build_model()

    assert isinstance(model, Pipeline)

    assert isinstance(
        model.named_steps["tfidf"],
        TfidfVectorizer,
    )

    assert isinstance(
        model.named_steps["classifier"],
        LogisticRegression,
    )


def test_model_can_fit_and_predict(sample_df):
    model = build_model()

    X = sample_df["text"]
    y = sample_df["label"]

    model.fit(X, y)

    predictions = model.predict(X.iloc[:3])

    assert len(predictions) == 3

    assert set(predictions).issubset(set(y))


def test_predict_does_not_refit_vectorizer(sample_df):
    model = build_model()

    model.fit(
        sample_df["text"],
        sample_df["label"],
    )

    vectorizer = model.named_steps["tfidf"]

    vocabulary_before = dict(vectorizer.vocabulary_)
    idf_before = vectorizer.idf_.copy()

    model.predict([
        "completelynewtoken appearing only during inference"
    ])

    assert vectorizer.vocabulary_ == vocabulary_before

    np.testing.assert_array_equal(
        vectorizer.idf_,
        idf_before,
    )
