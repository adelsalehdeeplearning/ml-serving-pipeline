# Use official Python image

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements (we’ll define it below)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model
COPY ./app ./app
COPY model.pt scaler.pt .

# Expose port
EXPOSE 80

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
