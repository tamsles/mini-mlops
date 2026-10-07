from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn

from mini_mlops.data import (
    load_data,
    split_data,
    validate_schema,
)
from mini_mlops.lineage import (
    get_git_commit,
    is_git_dirty,
    sha256_file,
)
from mini_mlops.model import (
    build_model,
    evaluate_model,
    train_model,
)
from mini_mlops.tracking import configure_tracking


@dataclass
class TrainingResult:
    run_id: str
    model_uri: str
    accuracy: float
    macro_f1: float
    git_commit: str
    git_dirty: bool
    data_sha256: str


def train_workload(
    data_path: Path,
    seed: int = 42,
    test_size: float = 0.2,
) -> TrainingResult:
    """Run the complete training workload."""

    # -------------------------
    # Data
    # -------------------------

    df = load_data(data_path)

    validate_schema(df)

    train_df, test_df = split_data(
        df=df,
        test_size=test_size,
        seed=seed,
    )

    # -------------------------
    # Lineage
    # -------------------------

    git_commit = get_git_commit()
    git_dirty = is_git_dirty()
    data_sha256 = sha256_file(data_path)

    # -------------------------
    # Tracking
    # -------------------------

    configure_tracking()

    with mlflow.start_run() as run:

        mlflow.log_params(
            {
                "seed": seed,
                "test_size": test_size,
                "data_path": str(data_path),
                "dataset_size": len(df),
                "train_size": len(train_df),
                "test_size_rows": len(test_df),
            }
        )

        mlflow.set_tags(
            {
                "task": "text-classification",
                "git_commit": git_commit,
                "git_dirty": str(git_dirty).lower(),
                "data_sha256": data_sha256,
            }
        )

        # -------------------------
        # Model
        # -------------------------

        model = build_model(
            seed=seed,
        )

        model = train_model(
            model=model,
            train_df=train_df,
        )

        metrics = evaluate_model(
            model=model,
            test_df=test_df,
        )

        mlflow.log_metrics(metrics)

        input_example = (
            train_df["text"]
            .head(3)
            .tolist()
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
        )

        return TrainingResult(
            run_id=run.info.run_id,
            model_uri=model_info.model_uri,
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
            git_commit=git_commit,
            git_dirty=git_dirty,
            data_sha256=data_sha256,
        )