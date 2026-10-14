
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = ROOT / "logs" / "predictions.jsonl"

_WRITE_LOCK = threading.Lock()


def write_event(
    *,
    log_path: Path,
    model_version: str | None,
    prediction: str | None,
    latency_ms: float,
    http_status: int,
    error_type: str | None,
    input_chars: int | None,
    input_words: int | None,
) -> None:
    """Append one prediction event as one JSON line."""

    event = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "event_id": str(uuid4()),
        "model_version": model_version,
        "prediction": prediction,
        "latency_ms": round(latency_ms, 3),
        "http_status": http_status,
        "error_type": error_type,
        "input_chars": input_chars,
        "input_words": input_words,
    }

    log_path = Path(log_path)

    with _WRITE_LOCK:
        log_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        with log_path.open(
            "a", encoding="utf-8"
        ) as file:
            file.write(
                json.dumps(event, ensure_ascii=False)
                + "\n"
            )


def install_monitoring(
    app: FastAPI,
    log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    """Install monitoring for the /predict endpoint."""

    @app.middleware("http")
    async def monitor_predict(request: Request, call_next):

        if request.url.path != "/predict":
            return await call_next(request)

        started = time.perf_counter()

        http_status = 500
        error_type = None

        try:
            response = await call_next(request)

            http_status = response.status_code

            if http_status >= 400:
                error_type = f"http_{http_status}"

            return response

        except Exception as exc:
            error_type = type(exc).__name__
            raise

        finally:
            latency_ms = (
                time.perf_counter() - started
            ) * 1000

            try:
                write_event(
                    log_path=log_path,
                    model_version=getattr(
                        request.app.state,
                        "model_version",
                        None,
                    ),
                    prediction=getattr(
                        request.state,
                        "prediction",
                        None,
                    ),
                    latency_ms=latency_ms,
                    http_status=http_status,
                    error_type=error_type,
                    input_chars=getattr(
                        request.state,
                        "input_chars",
                        None,
                    ),
                    input_words=getattr(
                        request.state,
                        "input_words",
                        None,
                    ),
                )

            except Exception as exc:
                print(
                    f"MONITORING WRITE ERROR: {exc}",
                    file=sys.stderr,
                )


def summarize(log_path: Path) -> dict:
    """Summarize JSONL events without external tools."""

    with Path(log_path).open(
        encoding="utf-8"
    ) as file:
        events = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    statuses = Counter(
        event["http_status"] for event in events
    )

    latencies = sorted(
        float(event["latency_ms"])
        for event in events
    )

    prediction_counts = defaultdict(Counter)

    for event in events:
        if (
            event["http_status"] < 400
            and event["prediction"] is not None
        ):
            version = (
                event["model_version"] or "unknown"
            )

            prediction_counts[version][
                event["prediction"]
            ] += 1

    n = len(events)

    chars = [
        event["input_chars"]
        for event in events
        if event["input_chars"] is not None
    ]

    client_errors = sum(
        count
        for code, count in statuses.items()
        if 400 <= code < 500
    )

    server_errors = sum(
        count
        for code, count in statuses.items()
        if code >= 500
    )

    return {
        "requests": n,
        "client_errors_4xx": client_errors,
        "server_errors_5xx": server_errors,
        "server_error_rate": (
            server_errors / n if n else 0.0
        ),
        "latency_p95_ms": (
            latencies[math.ceil(0.95 * n) - 1]
            if n else None
        ),
        "mean_input_chars": (
            round(sum(chars) / len(chars), 2)
            if chars else None
        ),
        "prediction_counts_by_model_version": {
            version: dict(counts)
            for version, counts
            in prediction_counts.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG_PATH,
    )

    args = parser.parse_args()

    print(
        json.dumps(
            summarize(args.log),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
