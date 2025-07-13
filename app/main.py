# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np

app = FastAPI()

# Define input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load model and scaler
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)
model.load_state_dict(torch.load("model.pt"))
model.eval()

scaler = torch.load("scaler.pt",weights_only=False)

@app.get("/")
def root():
    return {"message": "ML Model is ready!"}

@app.post("/predict")
def predict(data: IrisInput):
    try:
        features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
        features = scaler.transform(features)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            output = model(features_tensor)
            prediction = torch.argmax(output, dim=1).item()
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
