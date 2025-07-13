from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import requests

app = FastAPI()
path = 'D:/Downloaded_ws/app/templates'
templates = Jinja2Templates(directory=path)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float  
    petal_length: float
    petal_width: float

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)
model.load_state_dict(torch.load("model.pt"))
model.eval()

scaler = torch.load("scaler.pt", weights_only=False)

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

@app.get("/form", response_class=HTMLResponse)
def show_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "prediction": None})

@app.post("/predict-form", response_class=HTMLResponse)
def predict_from_form(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    # Use direct function call instead of HTTP request
    iris_data = IrisInput(
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_length=petal_length,
        petal_width=petal_width
    )
    result = predict(iris_data)
    prediction = result["prediction"]
    
    return templates.TemplateResponse("form.html", {"request": request, "prediction": prediction})