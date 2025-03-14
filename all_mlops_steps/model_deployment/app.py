import os
import joblib
import wandb
import logging
import pandas as pd
import random
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List
import requests

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# Define the Pydantic model for the prediction request
class PredictionRequest(BaseModel):
    amt: float
    hour: int
    day_of_week: int
    category: str
    state: str

    @field_validator('hour')
    def check_hour(cls, v):
        if v < 0 or v > 23:
            raise ValueError('Hour must be between 0 and 23')
        return v

    @field_validator('day_of_week')
    def check_day_of_week(cls, v):
        if v < 0 or v > 6:
            raise ValueError('Day of week must be between 0 and 6')
        return v
def load_artifacts():
    try:
        # Check if WANDB_API_KEY is set
        if not os.getenv('WANDB_API_KEY'):
            raise ValueError("WANDB_API_KEY is not set in the environment variables")

        # Check if WANDB_PROJECT is set
        if not os.getenv('WANDB_PROJECT'):
            raise ValueError("WANDB_PROJECT is not set in the environment variables")

        # Initialize WandB
        if hasattr(wandb, 'login'):
            wandb.login()
            logger.info("Successfully logged in to WandB")
        else:
            logger.warning("wandb.login() not available, continuing without login")

        # Start a new run
        if hasattr(wandb, 'init'):
            run = wandb.init(project=os.getenv('WANDB_PROJECT'), job_type="inference")
            logger.info(f"Initialized WandB run for project: {os.getenv('WANDB_PROJECT')}")
        else:
            logger.warning("wandb.init() not available, continuing without initializing a run")
            run = None

        # Load the model artifact
        artifact = run.use_artifact('production_model:latest', type='model')
        model_dir = artifact.download()
        model_path = f"{model_dir}/production_model.pkl"

        # Load the model using joblib
        model = joblib.load(model_path)
        print("Model loaded successfully")

        # Try loading the preprocessor artifact (use 'latest' or correct version)
        artifact = run.use_artifact('preprocessor:latest', type='model')
        pipeline_path = artifact.download()

        # Check if the file exists
        pipeline_file_path = f"{pipeline_path}/preprocessor.pkl"
        if not os.path.exists(pipeline_file_path):
            raise FileNotFoundError(f"Pipeline file not found at {pipeline_file_path}")

        # Load the preprocessor pipeline using joblib
        pipeline = joblib.load(pipeline_file_path)
        print("Pipeline loaded successfully")

        return model, pipeline

    except Exception as e:
        raise RuntimeError(f"Failed to load necessary artifacts: {str(e)}")


# Load model and pipeline
try:
    model, pipeline = load_artifacts()
    logger.info("Successfully loaded artifacts")
except Exception as e:
    logger.error(f"Failed to load artifacts: {str(e)}")


# Define the health check route
@app.get("/health")
async def health_check():
    """
    Endpoint for health checks.
    """
    return {"status": "healthy"}


# Define the prediction route
@app.post("/predict/")
async def predict(request: PredictionRequest):
    """
    Predict the output for the given input data using the loaded model and pipeline.
    """
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([request.model_dump()])
        # Apply preprocessing pipeline
        processed_data = pipeline.transform(data)

        # Make prediction
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[0][1]

        return {
            "prediction": int(prediction[0]),
            "fraud_probability": float(probability)
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Simulate user traffic for testing
BASE_URL = "http://localhost:9090"

def test_health_check():
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check status: {response.status_code}")
    print(f"Response: {response.json()}")

def send_prediction_request():
    data = {
        "amt": round(random.uniform(1, 1000), 2),
        "hour": random.randint(0, 23),
        "day_of_week": random.randint(0, 6),
        "category": random.choice(["grocery", "entertainment", "travel", "food"]),
        "state": random.choice(["CA", "NY", "TX", "FL"])
    }
    response = requests.post(f"{BASE_URL}/predict/", json=data)
    print(f"Prediction request status: {response.status_code}")
    print(f"Response: {response.json()}")

def simulate_user_traffic(num_requests=10, delay=1):
    print(f"Simulating {num_requests} user requests with {delay} second delay between requests")
    for i in range(num_requests):
        print(f"\nRequest {i+1}:")
        send_prediction_request()
        time.sleep(delay)

if __name__ == "__main__":
    # Test health check and simulate traffic
    test_health_check()
    print("\nStarting user traffic simulation...")
    simulate_user_traffic()

    # Start the FastAPI app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)
