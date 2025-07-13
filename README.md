
# ML Model Serving Pipeline

This project demonstrates how to serve a trained PyTorch model using FastAPI, Docker, and Kubernetes.

## 🔧 Tech Stack
- Python + PyTorch (ML model)
- FastAPI (API layer)
- Docker (containerization)
- Kubernetes (deployment via Minikube)
- GitHub (source control & CI/CD)

## 🚀 Features
- Train a simple ML model (Iris classifier)
- Serve predictions via FastAPI `/predict` endpoint
- Dockerize the API for portability
- Deploy to Kubernetes
- Supports future: CI/CD, monitoring, frontend, etc.

## 📦 Folder Structure
```
ml-serving-pipeline/
├── app/              # FastAPI app
├── model.pt          # Trained model
├── scaler.pt         # StandardScaler
├── train_model.py    # Training script
├── requirements.txt  # Python packages
├── Dockerfile        # Build image
├── k8s/              # K8s deployment YAMLs
│   ├── deployment.yaml
│   └── service.yaml
```

## 🔁 Run Locally
```bash
# Train the model
python train_model.py

# Build the image
docker build -t iris-fastapi-app .

# Run the container
docker run -p 8000:80 iris-fastapi-app

# Test
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}"
```

## ☸ Deploy to Kubernetes
```bash
kubectl apply -f k8s/
minikube service iris-model-service
```

## 📌 Author
[@adelsalehdeeplearning](https://github.com/adelsalehdeeplearning)
