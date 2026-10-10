
import mlflow.sklearn

from mini_mlops.promotion import (
    load_champion_model,
    read_champion,
)

champion = read_champion()

if champion is None:
    raise RuntimeError("Champion is not initialized")

model = load_champion_model()

model_version = champion["run_id"]

# # Load once when the service starts.
# model = mlflow.sklearn.load_model("serving_model")


def predict(text: str) -> str:
    return str(model.predict([text])[0])
