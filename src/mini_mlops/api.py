from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from mini_mlops import serving
from mini_mlops.monitoring import install_monitoring


app = FastAPI()

# Version of the ACTUALLY LOADED model.
app.state.model_version = serving.MODEL_VERSION

# Register middleware before the app starts serving.
install_monitoring(app)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)


@app.post("/predict")
def predict_endpoint(
    payload: PredictRequest,
    request: Request,
):
    # Data monitoring metadata.
    request.state.input_chars = len(payload.text)
    request.state.input_words = len(
        payload.text.split()
    )

    # Reuse existing inference logic.
    label = serving.predict(payload.text)

    # Model monitoring metadata.
    request.state.prediction = str(label)

    return {"label": str(label)}