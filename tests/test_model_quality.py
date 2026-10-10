
import pytest

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from mini_mlops.model import build_model

from mini_mlops.quality import (
    ModelQualityError,
    evaluate_quality,
    require_quality,
)


def test_perfect_predictions_pass():
    y_true = ["nlp", "cv", "mlops", "other"] * 3

    result = evaluate_quality(
        y_true,
        y_true,
        min_macro_f1=0.30,
    )

    assert result.macro_f1 == pytest.approx(1.0)
    assert result.passed

    require_quality(result)


def test_constant_predictions_are_rejected():
    y_true = ["nlp", "cv", "mlops", "other"] * 3
    y_pred = ["nlp"] * len(y_true)

    result = evaluate_quality(
        y_true,
        y_pred,
        min_macro_f1=0.30,
    )

    assert not result.passed

    with pytest.raises(ModelQualityError):
        require_quality(result)


def test_incomplete_evaluation_set_rejected():
    y_true = ["nlp", "cv", "mlops"]
    y_pred = ["nlp", "cv", "mlops"]

    with pytest.raises(
        ValueError,
        match="all four classes",
    ):
        evaluate_quality(y_true, y_pred)


def test_dummy_classifier_fails_gate(sample_df):
    dummy = DummyClassifier(strategy="most_frequent")

    dummy.fit(
        sample_df[["text"]],
        sample_df["label"],
    )

    predictions = dummy.predict(sample_df[["text"]])

    result = evaluate_quality(
        sample_df["label"],
        predictions,
        min_macro_f1=0.30,
    )

    assert not result.passed

    with pytest.raises(ModelQualityError):
        require_quality(result)


@pytest.mark.model_quality
def test_model_quality_on_holdout(sample_df):
    train_df, test_df = train_test_split(
        sample_df,
        test_size=0.33,
        random_state=42,
        stratify=sample_df["label"],
    )

    model = build_model()

    model.fit(
        train_df["text"],
        train_df["label"],
    )

    predictions = model.predict(test_df["text"])

    result = evaluate_quality(
        test_df["label"],
        predictions,
        min_macro_f1=0.30,
    )

    assert result.passed, (
        f"macro-F1={result.macro_f1:.3f}; "
        f"threshold={result.threshold:.3f}"
    )
