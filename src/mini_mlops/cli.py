from pathlib import Path

import mlflow
import mlflow.sklearn
import typer

from mini_mlops.data import (
    load_data,
    split_data,
    validate_schema,
)
from mini_mlops.model import (
    build_model,
    evaluate_model,
    train_model,
)
from mini_mlops.tracking import configure_tracking


app = typer.Typer()


@app.command()
def train(
    data_path: Path = Path("data/raw/projects.csv"),
    seed: int = 42,
    test_size: float = 0.2,
):
    """Train and track a text classifier."""

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
    # MLflow
    # -------------------------

    configure_tracking()

    with mlflow.start_run() as run:

        # -------------------------
        # Log configuration
        # -------------------------

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

        # -------------------------
        # Build + train
        # -------------------------

        model = build_model(
            seed=seed
        )

        model = train_model(
            model=model,
            train_df=train_df,
        )

        # -------------------------
        # Evaluate
        # -------------------------

        metrics = evaluate_model(
            model=model,
            test_df=test_df,
        )

        mlflow.log_metrics(metrics)

        # -------------------------
        # Log model artifact
        # -------------------------

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

        # -------------------------
        # Useful metadata
        # -------------------------

        mlflow.set_tag(
            "task",
            "text-classification",
        )

        # -------------------------
        # Console output
        # -------------------------

        print(f"Dataset size: {len(df)}")
        print(f"Train size: {len(train_df)}")
        print(f"Test size: {len(test_df)}")

        print("\nMetrics:")
        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )
        print(
            f"Macro F1: "
            f"{metrics['macro_f1']:.4f}"
        )

        print("\nMLflow:")
        print(
            f"Run ID: "
            f"{run.info.run_id}"
        )
        print(
            f"Model URI: "
            f"{model_info.model_uri}"
        )


@app.command()
def evaluate():
    """Evaluate a saved model."""
    print("Not implemented yet.")


if __name__ == "__main__":
    app()