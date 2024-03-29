from fastapi import FastAPI, UploadFile, File
from typing import List
import cv2
import numpy as np
import torch
import pandas as pd
import pathlib
import requests

import tempfile
import os
import mysql.connector
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(filename='myapp.log', level=logging.INFO)  # Main log file

logger = logging.getLogger(__name__)  # Default logger

db_logger = logging.getLogger("database")  # Separate logger for database
db_logger.setLevel(logging.DEBUG)  # More detailed database logging

requests_logger = logging.getLogger("requests")  # Logger for HTTP requests
requests_logger.setLevel(logging.INFO)  # Track requests and responses

model_logger = logging.getLogger("model")  # Logger for model-related events
model_logger.setLevel(logging.WARNING)  # Less verbose model logging


# Load environment variables
load_dotenv()

# Database connection
db = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_DATABASE")
)

cursor = db.cursor()

app = FastAPI()

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Model URL and download logic
url = "https://github.com/AbelBekele/ML-Microservice-Deployment/raw/main/model/best.pt"

def download_model():
    model_path = './models/best.pt'
    if not os.path.exists(model_path):
        try:
            response = requests.get(url, allow_redirects=True)
            response.raise_for_status()  # Raise error for non-2xx status codes

            pathlib.Path('./models').mkdir(parents=True, exist_ok=True)

            with open(model_path, 'wb') as file:
                file.write(response.content)

            requests_logger.info(f"Model downloaded successfully to: {model_path}")
        except requests.exceptions.RequestException as e:
            requests_logger.error(f"Failed to download model: {e}")
            raise Exception("Failed to download model")

# Load the model (can be called within a dependency or function)
def load_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', './models/best.pt', _verbose=False)
        model.conf = 0.7  # NMS confidence threshold
        model_logger.info(f"Model loaded successfully from path: ./models/best.pt")
        return model
    except Exception as e:
        model_logger.error(f"Failed to load model: {e}")
        raise Exception("Failed to load model")

# Class for image data type
class ImageType(UploadFile):
    data: bytes


@app.post("/predict")
async def predict(image: UploadFile):
    """
    Predicts objects in an uploaded image.

    Raises:
        Exception: If model download or loading fails.
    """

    try:
        # Download model if it doesn't exist
        download_model()

        # Load the model
        model = load_model()

        # Read the image file
        image_bytes = await image.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Inference
        results = model([img], size=416)

        # Get pandas DataFrame of results
        results_df = results.pandas().xyxy[0]

        # Count occurrences of each class name
        counts = results_df['name'].value_counts().to_dict()

        # Convert counts to string
        counts_str = ', '.join(f'{v} {k}' for k, v in counts.items())

        return {"results": counts_str}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"error": str(e)}


@app.get("/health")
async def health_check():
    return {"status": "UP"}

# Run the FastAPI application locally
if __name__ == "__main__":
 import uvicorn
 uvicorn.run(app, host="0.0.0.0", port=8000)