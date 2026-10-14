
import mlflow.sklearn

from mini_mlops.promotion import (
    load_champion_model,
    read_champion,
)

champion = read_champion()

if champion is None:
    raise RuntimeError(
        "No champion has been promoted"
    )

mlflow.set_tracking_uri(
    champion["tracking_uri"]
)

model = mlflow.sklearn.load_model(
    champion["model_uri"]
)

MODEL_VERSION = champion["run_id"]

# # Load once when the service starts.
# model = mlflow.sklearn.load_model("serving_model")


def predict(text: str) -> str:
    return str(model.predict([text])[0])
