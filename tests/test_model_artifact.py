
import mlflow.sklearn
import numpy as np
import pytest

from sklearn.pipeline import Pipeline


@pytest.mark.integration
def test_model_artifact_roundtrip(
    fitted_model,
    tmp_path,
):
    model_path = tmp_path / "saved_model"

    inputs = [
        "BERT transformer language model",
        "model monitoring and deployment",
        "image recognition using convolutional networks",
    ]

    predictions_before = fitted_model.predict(inputs)

    # Save the complete sklearn Pipeline.
    mlflow.sklearn.save_model(
        sk_model=fitted_model,
        path=str(model_path),
        serialization_format="cloudpickle",
    )

    # Load the model from the artifact.
    restored_model = mlflow.sklearn.load_model(
        str(model_path)
    )

    predictions_after = restored_model.predict(inputs)

    # Model type and structure
    assert isinstance(restored_model, Pipeline)

    assert "tfidf" in restored_model.named_steps
    assert "classifier" in restored_model.named_steps

    # Prediction equivalence
    np.testing.assert_array_equal(
        predictions_before,
        predictions_after,
    )

    # Preprocessing state equivalence
    original_tfidf = fitted_model.named_steps["tfidf"]
    restored_tfidf = restored_model.named_steps["tfidf"]

    assert (
        original_tfidf.vocabulary_
        == restored_tfidf.vocabulary_
    )

    np.testing.assert_array_equal(
        original_tfidf.idf_,
        restored_tfidf.idf_,
    )
