from pathlib import Path
import typer
from mini_mlops.training import train_workload

app = typer.Typer()


@app.command()
def train(
    data_path: Path = Path("data/raw/projects.csv"),
    seed: int = 42,
    test_size: float = 0.2,
):
    """Train and track a text classifier."""

    result = train_workload(
        data_path=data_path,
        seed=seed,
        test_size=test_size,
    )

    print("\nTraining completed")

    print(f"Run ID: {result.run_id}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Macro F1: {result.macro_f1:.4f}")

    print("\nLineage:")
    print(f"Git commit: {result.git_commit}")
    print(f"Git dirty: {result.git_dirty}")
    print(f"Data SHA256: {result.data_sha256}")

    print("\nArtifact:")
    print(f"Model URI: {result.model_uri}")



@app.command()
def evaluate():
    """Evaluate a saved model."""
    print("Not implemented yet.")


if __name__ == "__main__":
    app()