
import mlflow.sklearn

# Load once when the service starts.
model = mlflow.sklearn.load_model("serving_model")


def predict(text: str) -> str:
    return str(model.predict([text])[0])
