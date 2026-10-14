
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd

from mini_mlops.contracts import validate_training_data
from mini_mlops.drift import analyze
from mini_mlops.lineage import sha256_file
from mini_mlops.promotion import (
    PromotionRejected,
    promote_candidate,
    read_champion,
)
from mini_mlops.training import train_workload


ROOT = Path(__file__).resolve().parents[2]


def _run_tests() -> None:
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=ROOT,
        check=True,
    )


def run_feedback_loop(
    *,
    reference_path: Path,
    log_path: Path,
    champion_path: Path,
    training_data: Path | None = None,
    execute: bool = False,
    min_events: int = 10,
    input_threshold: float = 0.50,
    prediction_threshold: float = 0.40,
    quality_threshold: float = 0.60,
    seed: int = 42,
    test_size: float = 0.20,
) -> dict:
    """Execute one feedback-loop iteration."""

    if (
        not math.isfinite(quality_threshold)
        or not 0 <= quality_threshold <= 1
    ):
        raise ValueError(
            "quality_threshold must be in [0, 1]"
        )

    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")

    # Read the currently selected production model.
    champion_before = read_champion(champion_path)

    if champion_before is None:
        raise RuntimeError("Champion is not initialized")

    for key in ("run_id", "model_uri", "tracking_uri"):
        if not champion_before.get(key):
            raise RuntimeError(f"Champion missing {key}")

    mlflow.set_tracking_uri(
        champion_before["tracking_uri"]
    )

    # 1. Analyze production monitoring events.
    drift = analyze(
        reference_path=reference_path,
        log_path=log_path,
        champion_path=champion_path,
        min_events=min_events,
        input_threshold=input_threshold,
        prediction_threshold=prediction_threshold,
    )

    info = {
        "champion_before": champion_before["run_id"],
        "drift": drift,
    }

    # 2. Decide whether retraining is warranted.
    if drift["status"] == "INSUFFICIENT_DATA":
        return {
            "status": "INSUFFICIENT_DATA",
            **info,
        }

    if drift["status"] == "OK":
        return {
            "status": "SKIPPED_NO_DRIFT",
            **info,
        }

    if drift["status"] != "WARNING":
        raise RuntimeError(
            f"Unknown drift status: {drift['status']}"
        )

    # Safe default: warning alone cannot train/promote.
    if not execute:
        return {
            "status": "DRIFT_WARNING_ONLY",
            **info,
        }

    if training_data is None:
        raise ValueError(
            "--training-data is required with --execute on WARNING"
        )

    training_data = Path(training_data).resolve()

    # 3. Validate labeled training data.
    df = pd.read_csv(training_data)
    validate_training_data(df)

    expected_sha = sha256_file(training_data)

    # 4. Block training when code tests fail.
    _run_tests()

    if sha256_file(training_data) != expected_sha:
        raise RuntimeError(
            "Training data changed after validation"
        )

    # 5. Produce a NEW candidate using existing workload.
    candidate = train_workload(
        data_path=training_data,
        seed=seed,
        test_size=test_size,
    )

    if not candidate.run_id or not candidate.model_uri:
        raise RuntimeError(
            "Training produced no candidate identity"
        )

    if (
        candidate.data_sha256 != expected_sha
        or sha256_file(training_data) != expected_sha
    ):
        raise RuntimeError(
            "Candidate trained on different data"
        )

    # Stop if champion changed while we were training.
    if read_champion(champion_path) != champion_before:
        raise RuntimeError(
            "Champion changed during retraining; abort"
        )

    candidate_info = {
        "candidate_run_id": candidate.run_id,
        "candidate_model_uri": candidate.model_uri,
    }

    # 6. Promotion must read candidate's metric from MLflow.
    mlflow.set_tracking_uri(
        champion_before["tracking_uri"]
    )

    try:
        promoted = promote_candidate(
            run_id=candidate.run_id,
            model_uri=candidate.model_uri,
            threshold=quality_threshold,
            path=champion_path,
        )

    except PromotionRejected as exc:
        # A rejection must NOT mutate the champion.
        if read_champion(champion_path) != champion_before:
            raise RuntimeError(
                "Rejected candidate changed champion"
            ) from exc

        return {
            "status": "REJECTED",
            **info,
            **candidate_info,
            "reason": str(exc),
            "champion_after": champion_before["run_id"],
        }

    # 7. Verify the selected model really changed.
    current = read_champion(champion_path)

    if (
        promoted.get("run_id") != candidate.run_id
        or current is None
        or current.get("run_id") != candidate.run_id
        or current.get("model_uri") != candidate.model_uri
    ):
        raise RuntimeError(
            "Promotion returned inconsistent champion state"
        )

    return {
        "status": "PROMOTED",
        **info,
        **candidate_info,
        "champion_after": current["run_id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One safe retraining iteration"
    )

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
        "--training-data",
        type=Path,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
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
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
    )

    args = parser.parse_args()
    os.chdir(ROOT)

    try:
        result = run_feedback_loop(
            reference_path=args.reference,
            log_path=args.log,
            champion_path=args.champion,
            training_data=args.training_data,
            execute=args.execute,
            min_events=args.min_events,
            input_threshold=args.input_threshold,
            prediction_threshold=args.prediction_threshold,
            quality_threshold=args.quality_threshold,
            seed=args.seed,
            test_size=args.test_size,
        )

    except Exception as exc:
        print(
            f"FEEDBACK ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    return {
        "SKIPPED_NO_DRIFT": 0,
        "DRIFT_WARNING_ONLY": 10,
        "INSUFFICIENT_DATA": 11,
        "REJECTED": 20,
        "PROMOTED": 0,
    }[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
