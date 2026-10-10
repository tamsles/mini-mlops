
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn

from mini_mlops.quality import (
    ModelQualityResult,
    require_quality,
)


ROOT = Path(__file__).resolve().parents[2]
CHAMPION_PATH = ROOT / "state" / "champion.json"

ALLOWED_LABELS = {"nlp", "cv", "mlops", "other"}


class PromotionRejected(Exception):
    """Candidate does not satisfy promotion requirements."""


def read_champion(path: Path = CHAMPION_PATH) -> dict | None:
    """Return champion metadata, or None if not initialized."""
    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def _source_run_id(model_uri: str) -> str | None:
    """Resolve the source training run of an MLflow model."""

    # MLflow 2-style run artifact URI.
    if model_uri.startswith("runs:/"):
        parts = model_uri.removeprefix("runs:/").split("/", 1)
        if len(parts) != 2 or not all(parts):
            raise PromotionRejected("Invalid runs:/ model URI")
        return parts[0]

    # MLflow 3 logged-model URI, e.g. models:/m-xxxx.
    if model_uri.startswith("models:/m-"):
        model_id = model_uri.removeprefix("models:/")
        logged_model = mlflow.get_logged_model(model_id)
        return logged_model.source_run_id

    # Other supported MLflow model references.
    info = mlflow.models.get_model_info(model_uri)
    return info.run_id


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Replace champion pointer only after a complete write."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".champion-",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            json.dump(payload, file, indent=2)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, path)

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def promote_candidate(
    run_id: str,
    model_uri: str,
    threshold: float = 0.60,
    path: Path = CHAMPION_PATH,
) -> dict:
    """
    Verify one candidate and update the champion pointer.

    On rejection, leave the existing champion unchanged.
    """
    path = Path(path)

    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be within [0, 1]")

    if not run_id or not model_uri:
        raise PromotionRejected("Missing run_id or model_uri")

    # 1. Retrieve the actual recorded training run.
    run = mlflow.get_run(run_id)

    if run.info.status != "FINISHED":
        raise PromotionRejected(
            f"Run is not finished: {run.info.status}"
        )

    # 2. Obtain the metric from MLflow, not caller input.
    score = run.data.metrics.get("macro_f1")

    if score is None or not math.isfinite(score):
        raise PromotionRejected("Missing or invalid macro_f1")

    # 3. Reuse the Day 2 quality gate.
    passed = 0 <= score <= 1 and score >= threshold

    quality = ModelQualityResult(
        macro_f1=score,
        threshold=threshold,
        passed=passed,
    )

    try:
        require_quality(quality)
    except Exception as exc:
        raise PromotionRejected(
            f"QUALITY REJECT: macro_f1={score:.4f}, "
            f"threshold={threshold:.4f}"
        ) from exc

    if not passed:
        raise PromotionRejected(
            f"QUALITY REJECT: macro_f1={score:.4f}"
        )

    # 4. Verify artifact belongs to this training run.
    source_run_id = _source_run_id(model_uri)

    if source_run_id != run_id:
        raise PromotionRejected(
            "Model artifact does not belong to candidate run"
        )

    # 5. Load the artifact and run a minimal inference check.
    model = mlflow.sklearn.load_model(model_uri)

    samples = [
        "BERT transformer language model",
        "computer vision image classification",
    ]
    predictions = model.predict(samples)

    if (
        len(predictions) != len(samples)
        or any(str(p) not in ALLOWED_LABELS for p in predictions)
    ):
        raise PromotionRejected(
            "Model artifact failed inference smoke test"
        )

    # 6. Build the deployment record.
    record = {
        "run_id": run_id,
        "model_uri": model_uri,
        "macro_f1": float(score),
        "tracking_uri": mlflow.get_tracking_uri(),
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }

    # The ONLY promotion state mutation.
    _atomic_write_json(path, record)

    return record


def load_champion_model(
    path: Path = CHAMPION_PATH,
):
    """Load the model selected by the current champion pointer."""
    champion = read_champion(path)

    if champion is None:
        raise RuntimeError("No champion has been promoted")

    mlflow.set_tracking_uri(champion["tracking_uri"])

    return mlflow.sklearn.load_model(champion["model_uri"])


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-uri", required=True)
    parser.add_argument(
        "--threshold", type=float, default=0.60
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get(
            "MLFLOW_TRACKING_URI",
            "sqlite:///mlflow.db",
        ),
    )

    args = parser.parse_args()

    os.chdir(ROOT)
    mlflow.set_tracking_uri(args.tracking_uri)

    try:
        record = promote_candidate(
            run_id=args.run_id,
            model_uri=args.model_uri,
            threshold=args.threshold,
        )

    except Exception as exc:
        print(
            f"PROMOTION FAILED: {type(exc).__name__}: {exc}"
        )
        return 1

    print("PROMOTION SUCCESS")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
