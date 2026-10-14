
import json

import pytest

from mini_mlops.drift import (
    evaluate_drift,
    ks_distance,
    read_current_events,
    tv_distance,
)


def test_ks_identical_distribution():
    a = [10, 20, 30, 40]
    b = [10, 20, 30, 40]

    assert ks_distance(a, b) == 0.0


def test_ks_detects_complete_shift():
    reference = [10, 20, 30, 40]
    current = [110, 120, 130, 140]

    assert ks_distance(reference, current) == 1.0


def test_prediction_tv_distance():
    reference = ["nlp", "cv", "mlops", "other"]
    current = ["nlp", "nlp", "nlp", "nlp"]

    assert tv_distance(reference, current) == pytest.approx(
        0.75
    )


def test_input_shift_without_prediction_shift():
    reference = ["nlp", "cv", "mlops", "other"]

    result = evaluate_drift(
        reference_chars=[10, 20, 30, 40],
        current_chars=[110, 120, 130, 140],
        reference_words=[1, 2, 3, 4],
        current_words=[1, 2, 3, 4],
        reference_predictions=reference,
        current_predictions=reference,
    )

    assert result["status"] == "WARNING"
    assert result["input_chars_ks"] == 1.0
    assert result["prediction_tv"] == 0.0

    assert result["warnings"] == [
        "input_chars_drift"
    ]


def test_prediction_shift_without_input_shift():
    reference = ["nlp", "cv", "mlops", "other"]

    result = evaluate_drift(
        reference_chars=[10, 20, 30, 40],
        current_chars=[10, 20, 30, 40],
        reference_words=[1, 2, 3, 4],
        current_words=[1, 2, 3, 4],
        reference_predictions=reference,
        current_predictions=["nlp"] * 4,
    )

    assert result["status"] == "WARNING"
    assert result["input_chars_ks"] == 0.0
    assert result["input_words_ks"] == 0.0
    assert result["prediction_tv"] == 0.75

    assert result["warnings"] == [
        "prediction_drift"
    ]


def test_filters_model_version_and_failed_requests(
    tmp_path,
):
    path = tmp_path / "events.jsonl"

    events = [
        {
            "model_version": "champion-A",
            "http_status": 200,
            "prediction": "nlp",
            "input_chars": 20,
            "input_words": 4,
        },
        {
            "model_version": "champion-B",
            "http_status": 200,
            "prediction": "cv",
            "input_chars": 20,
            "input_words": 4,
        },
        {
            "model_version": "champion-A",
            "http_status": 422,
            "prediction": None,
            "input_chars": None,
            "input_words": None,
        },
    ]

    path.write_text(
        "\n".join(json.dumps(e) for e in events) + "\n",
        encoding="utf-8",
    )

    result = read_current_events(
        path,
        model_version="champion-A",
    )

    assert result["total_events"] == 3
    assert result["matched_version_events"] == 2
    assert result["valid_prediction_events"] == 1

    assert result["observations"][0]["prediction"] == "nlp"
