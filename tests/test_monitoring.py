
import json

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from mini_mlops.monitoring import (
    install_monitoring,
    summarize,
)


class Payload(BaseModel):
    text: str = Field(min_length=1)


def make_app(log_path):
    app = FastAPI()

    app.state.model_version = "run-test-001"

    install_monitoring(
        app,
        log_path=log_path,
    )

    @app.post("/predict")
    def predict(payload: Payload, request: Request):

        request.state.input_chars = len(
            payload.text
        )

        request.state.input_words = len(
            payload.text.split()
        )

        if payload.text == "explode":
            raise RuntimeError(
                "simulated inference error"
            )

        request.state.prediction = "nlp"

        return {"label": "nlp"}

    return app


def test_logs_success_without_raw_text(tmp_path):
    path = tmp_path / "predictions.jsonl"

    with TestClient(make_app(path)) as client:
        result = client.post(
            "/predict",
            json={"text": "secret document"},
        )

    assert result.status_code == 200

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1

    event = json.loads(lines[0])

    assert event["model_version"] == "run-test-001"
    assert event["prediction"] == "nlp"
    assert event["http_status"] == 200
    assert event["input_chars"] == len(
        "secret document"
    )
    assert event["latency_ms"] >= 0

    # Never store original request text.
    assert "secret document" not in lines[0]


def test_validation_failure_logs_422(tmp_path):
    path = tmp_path / "predictions.jsonl"

    with TestClient(make_app(path)) as client:
        result = client.post(
            "/predict",
            json={"text": ""},
        )

    assert result.status_code == 422

    event = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert event["http_status"] == 422
    assert event["error_type"] == "http_422"
    assert event["prediction"] is None
    assert event["input_chars"] is None


def test_inference_error_logs_500(tmp_path):
    path = tmp_path / "predictions.jsonl"

    with TestClient(
        make_app(path),
        raise_server_exceptions=False,
    ) as client:
        result = client.post(
            "/predict",
            json={"text": "explode"},
        )

    assert result.status_code == 500

    event = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert event["http_status"] == 500
    assert event["error_type"] == "RuntimeError"
    assert event["prediction"] is None


def test_log_failure_does_not_break_inference(
    tmp_path,
):
    # A directory cannot be opened as a normal log file.
    # This intentionally makes the logger fail.
    with TestClient(make_app(tmp_path)) as client:
        result = client.post(
            "/predict",
            json={"text": "hello"},
        )

    # Logging failure must not turn 200 into 500.
    assert result.status_code == 200


def test_summary(tmp_path):
    path = tmp_path / "predictions.jsonl"

    with TestClient(
        make_app(path),
        raise_server_exceptions=False,
    ) as client:
        client.post(
            "/predict",
            json={"text": "hello"},
        )

        client.post(
            "/predict",
            json={"text": ""},
        )

        client.post(
            "/predict",
            json={"text": "explode"},
        )

    summary = summarize(path)

    assert summary["requests"] == 3
    assert summary["client_errors_4xx"] == 1
    assert summary["server_errors_5xx"] == 1

    assert (
        summary["prediction_counts_by_model_version"]
        ["run-test-001"]["nlp"]
        == 1
    )
