
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

LABELS = ("nlp", "cv", "mlops", "other")


def ks_distance(reference, current) -> float:
    """Two-sample empirical KS distance."""

    if not reference or not current:
        raise ValueError("KS requires nonempty samples")

    a = sorted(reference)
    b = sorted(current)

    if not all(math.isfinite(x) for x in a + b):
        raise ValueError("Nonfinite numeric observation")

    values = sorted(set(a + b))

    return max(
        abs(
            bisect_right(a, x) / len(a)
            - bisect_right(b, x) / len(b)
        )
        for x in values
    )


def tv_distance(reference, current) -> float:
    """Total variation distance for prediction labels."""

    if not reference or not current:
        raise ValueError("TVD requires nonempty samples")

    if any(x not in LABELS for x in reference + current):
        raise ValueError("Unexpected prediction label")

    a = Counter(reference)
    b = Counter(current)

    return 0.5 * sum(
        abs(
            a[label] / len(reference)
            - b[label] / len(current)
        )
        for label in LABELS
    )


def label_distribution(labels) -> dict:
    counts = Counter(labels)
    n = len(labels)

    return {
        label: round(counts[label] / n, 4)
        for label in LABELS
    }


def evaluate_drift(
    *,
    reference_chars,
    current_chars,
    reference_words,
    current_words,
    reference_predictions,
    current_predictions,
    input_threshold=0.50,
    prediction_threshold=0.40,
) -> dict:
    """Compare observable input and prediction distributions."""

    for threshold in (input_threshold, prediction_threshold):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must be in [0, 1]")

    ref_n = len(reference_chars)
    cur_n = len(current_chars)

    if not (
        ref_n == len(reference_words)
        == len(reference_predictions)
    ):
        raise ValueError("Reference arrays have unequal lengths")

    if not (
        cur_n == len(current_words)
        == len(current_predictions)
    ):
        raise ValueError("Current arrays have unequal lengths")

    chars_ks = ks_distance(reference_chars, current_chars)
    words_ks = ks_distance(reference_words, current_words)

    prediction_tv = tv_distance(
        list(reference_predictions),
        list(current_predictions),
    )

    warnings = []

    if chars_ks > input_threshold:
        warnings.append("input_chars_drift")

    if words_ks > input_threshold:
        warnings.append("input_words_drift")

    if prediction_tv > prediction_threshold:
        warnings.append("prediction_drift")

    return {
        "status": "WARNING" if warnings else "OK",
        "reference_n": ref_n,
        "current_n": cur_n,
        "input_chars_ks": round(chars_ks, 4),
        "input_words_ks": round(words_ks, 4),
        "prediction_tv": round(prediction_tv, 4),
        "reference_prediction_distribution":
            label_distribution(reference_predictions),
        "current_prediction_distribution":
            label_distribution(current_predictions),
        "warnings": warnings,
    }


def read_current_events(
    log_path: Path,
    model_version: str,
) -> dict:
    """Select valid successful predictions for one model version."""

    observations = []
    total_events = 0
    matched_version_events = 0

    with Path(log_path).open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue

            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError(
                    f"Invalid event at line {line_no}"
                )

            total_events += 1

            # Never mix predictions from two model versions.
            if event.get("model_version") != model_version:
                continue

            matched_version_events += 1

            # Exclude validation failures and inference errors.
            if event.get("http_status") != 200:
                continue

            chars = event.get("input_chars")
            words = event.get("input_words")
            label = event.get("prediction")

            # Missing telemetry is an error, not evidence of no drift.
            if (
                type(chars) is not int
                or type(words) is not int
                or chars < 0
                or words < 0
                or label not in LABELS
            ):
                raise ValueError(
                    f"Invalid successful event at line {line_no}"
                )

            observations.append({
                "input_chars": chars,
                "input_words": words,
                "prediction": label,
            })

    return {
        "observations": observations,
        "total_events": total_events,
        "matched_version_events": matched_version_events,
        "valid_prediction_events": len(observations),
    }


def analyze(
    *,
    reference_path: Path,
    log_path: Path,
    champion_path: Path,
    min_events: int = 10,
    input_threshold: float = 0.50,
    prediction_threshold: float = 0.40,
) -> dict:
    """Analyze current monitoring logs against a reference."""

    if min_events < 1:
        raise ValueError("min_events must be >= 1")

    champion = json.loads(
        Path(champion_path).read_text(encoding="utf-8")
    )

    model_version = champion["run_id"]
    model_uri = champion["model_uri"]

    current = read_current_events(
        log_path,
        model_version,
    )

    events = current["observations"]

    metadata = {
        "model_version": model_version,
        "total_log_events": current["total_events"],
        "matched_version_events":
            current["matched_version_events"],
        "valid_prediction_events":
            current["valid_prediction_events"],
    }

    # Insufficient data is not the same as "no drift".
    if len(events) < min_events:
        return {
            "status": "INSUFFICIENT_DATA",
            **metadata,
            "min_events": min_events,
            "warnings": [],
            "reason": "Not enough valid events for this model version",
        }

    df = pd.read_csv(reference_path)

    if "text" not in df.columns:
        raise ValueError("Reference CSV needs a text column")

    if df["text"].isna().any():
        raise ValueError("Reference contains missing text")

    texts = df["text"].astype(str).tolist()

    if not texts or any(not t.strip() for t in texts):
        raise ValueError("Reference contains empty text")

    # Use exactly the champion model version selected above.
    mlflow.set_tracking_uri(champion["tracking_uri"])
    model = mlflow.sklearn.load_model(model_uri)

    reference_predictions = [
        str(p) for p in model.predict(texts)
    ]

    current_predictions = [
        e["prediction"] for e in events
    ]

    stats = evaluate_drift(
        reference_chars=[len(t) for t in texts],
        current_chars=[e["input_chars"] for e in events],
        reference_words=[len(t.split()) for t in texts],
        current_words=[e["input_words"] for e in events],
        reference_predictions=reference_predictions,
        current_predictions=current_predictions,
        input_threshold=input_threshold,
        prediction_threshold=prediction_threshold,
    )

    reference_sha = hashlib.sha256(
        Path(reference_path).read_bytes()
    ).hexdigest()

    return {
        **metadata,
        "reference_sha256": reference_sha,
        "thresholds": {
            "input_ks": input_threshold,
            "prediction_tv": prediction_threshold,
        },
        **stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/raw/projects.csv"),
    )

    parser.add_argument(
        "--log",
        type=Path,
        default=Path("logs/predictions.jsonl"),
    )

    parser.add_argument(
        "--champion",
        type=Path,
        default=Path("state/champion.json"),
    )

    parser.add_argument(
        "--min-events",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--input-threshold",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=0.40,
    )

    args = parser.parse_args()

    os.chdir(ROOT)

    try:
        result = analyze(
            reference_path=args.reference,
            log_path=args.log,
            champion_path=args.champion,
            min_events=args.min_events,
            input_threshold=args.input_threshold,
            prediction_threshold=args.prediction_threshold,
        )

    except Exception as exc:
        print(
            f"DRIFT ANALYSIS FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result["status"] == "WARNING":
        return 10

    if result["status"] == "INSUFFICIENT_DATA":
        return 11

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
