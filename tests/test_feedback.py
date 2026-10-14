
import json
import subprocess
from types import SimpleNamespace

import pytest

from mini_mlops import feedback, promotion
from mini_mlops.lineage import sha256_file


@pytest.fixture
def example(tmp_path):
    pointer = tmp_path / "champion.json"

    champion = {
        "run_id": "old-run",
        "model_uri": "runs:/old-run/model",
        "tracking_uri": "sqlite:///mlflow.db",
    }

    pointer.write_text(
        json.dumps(champion),
        encoding="utf-8",
    )

    training_data = tmp_path / "training.csv"
    training_data.write_text(
        "text,label\nhello,nlp\n",
        encoding="utf-8",
    )

    return pointer, champion, training_data


def run_once(example, **kwargs):
    pointer, _, training_data = example

    return feedback.run_feedback_loop(
        reference_path=training_data,
        log_path=training_data,
        champion_path=pointer,
        training_data=training_data,
        **kwargs,
    )


def configure_warning_training(example, monkeypatch):
    _, _, training_data = example

    # Force a drift warning.
    monkeypatch.setattr(
        feedback,
        "analyze",
        lambda **kwargs: {
            "status": "WARNING",
            "warnings": ["input_chars_drift"],
        },
    )

    # Isolate orchestration from contracts and pytest.
    monkeypatch.setattr(
        feedback,
        "validate_training_data",
        lambda df: None,
    )

    monkeypatch.setattr(
        feedback,
        "_run_tests",
        lambda: None,
    )

    # Simulate a completed training workload.
    monkeypatch.setattr(
        feedback,
        "train_workload",
        lambda **kwargs: SimpleNamespace(
            run_id="candidate-run",
            model_uri="runs:/candidate-run/model",
            data_sha256=sha256_file(training_data),
        ),
    )


def test_no_drift_never_trains(example, monkeypatch):
    monkeypatch.setattr(
        feedback,
        "analyze",
        lambda **kwargs: {
            "status": "OK",
            "warnings": [],
        },
    )

    monkeypatch.setattr(
        feedback,
        "train_workload",
        lambda **kwargs: pytest.fail(
            "Training must not run"
        ),
    )

    result = run_once(example, execute=True)

    assert result["status"] == "SKIPPED_NO_DRIFT"
    assert json.loads(
        example[0].read_text()
    ) == example[1]


def test_warning_is_dry_run_by_default(
    example, monkeypatch
):
    configure_warning_training(example, monkeypatch)

    monkeypatch.setattr(
        feedback,
        "train_workload",
        lambda **kwargs: pytest.fail(
            "Training must not run"
        ),
    )

    result = run_once(example, execute=False)

    assert result["status"] == "DRIFT_WARNING_ONLY"
    assert json.loads(
        example[0].read_text()
    ) == example[1]


def test_insufficient_data_does_not_train(
    example, monkeypatch
):
    monkeypatch.setattr(
        feedback,
        "analyze",
        lambda **kwargs: {
            "status": "INSUFFICIENT_DATA",
            "warnings": [],
        },
    )

    monkeypatch.setattr(
        feedback,
        "train_workload",
        lambda **kwargs: pytest.fail(
            "Training must not run"
        ),
    )

    result = run_once(example, execute=True)

    assert result["status"] == "INSUFFICIENT_DATA"


def test_bad_candidate_cannot_replace_champion(
    example, monkeypatch
):
    configure_warning_training(example, monkeypatch)

    # Keep the REAL promotion logic.
    # Only replace the MLflow run lookup.
    monkeypatch.setattr(
        promotion.mlflow,
        "get_run",
        lambda run_id: SimpleNamespace(
            info=SimpleNamespace(status="FINISHED"),
            data=SimpleNamespace(
                metrics={"macro_f1": 0.20}
            ),
        ),
    )

    result = run_once(
        example,
        execute=True,
        quality_threshold=0.60,
    )

    assert result["status"] == "REJECTED"
    assert result["candidate_run_id"] == "candidate-run"
    assert result["champion_after"] == "old-run"

    actual = json.loads(
        example[0].read_text(encoding="utf-8")
    )

    assert actual == example[1]


def test_passed_candidate_changes_pointer(
    example, monkeypatch
):
    configure_warning_training(example, monkeypatch)

    # Isolate the orchestration success path.
    # Step 3 separately tests the real promotion gate.
    def fake_promote(
        *, run_id, model_uri, threshold, path
    ):
        record = {
            "run_id": run_id,
            "model_uri": model_uri,
            "tracking_uri": "sqlite:///mlflow.db",
        }

        path.write_text(
            json.dumps(record),
            encoding="utf-8",
        )

        return record

    monkeypatch.setattr(
        feedback,
        "promote_candidate",
        fake_promote,
    )

    result = run_once(example, execute=True)

    assert result["status"] == "PROMOTED"
    assert result["champion_after"] == "candidate-run"

    actual = json.loads(
        example[0].read_text()
    )

    assert actual["run_id"] == "candidate-run"


def test_pytest_failure_stops_training(
    example, monkeypatch
):
    configure_warning_training(example, monkeypatch)

    def fail_tests():
        raise subprocess.CalledProcessError(
            1, ["pytest"]
        )

    monkeypatch.setattr(
        feedback,
        "_run_tests",
        fail_tests,
    )

    monkeypatch.setattr(
        feedback,
        "train_workload",
        lambda **kwargs: pytest.fail(
            "Training must not run"
        ),
    )

    with pytest.raises(subprocess.CalledProcessError):
        run_once(example, execute=True)

    actual = json.loads(
        example[0].read_text()
    )

    assert actual == example[1]
