
from fastapi import FastAPI
from pydantic import BaseModel, Field

from mini_mlops.serving import predict


app = FastAPI()


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)


@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    label = predict(request.text)
    return {"label": label}
