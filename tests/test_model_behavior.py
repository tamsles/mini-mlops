
import pytest


GOLDEN_CASES = [
    ("BERT transformer language model", "nlp"),
    ("model monitoring and deployment", "mlops"),
    (
        "image recognition using convolutional networks",
        "cv",
    ),
    (
        "database indexing and operating system performance",
        "other",
    ),
]


@pytest.mark.model_quality
@pytest.mark.parametrize(
    "text, expected_label",
    GOLDEN_CASES,
)
def test_golden_predictions(
    fitted_model,
    text,
    expected_label,
):
    prediction = fitted_model.predict([text])[0]

    assert prediction == expected_label, (
        f"Input: {text!r}\n"
        f"Expected: {expected_label}\n"
        f"Actual: {prediction}"
    )
