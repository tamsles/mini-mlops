import mlflow


TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "mini-mlops"


def configure_tracking() -> None:
    """Configure the local MLflow tracking backend."""

    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )