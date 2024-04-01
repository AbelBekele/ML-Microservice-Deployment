from fastapi import FastAPI, UploadFile, File
from typing import List
import cv2
import numpy as np
import torch
import pandas as pd
import pathlib
import requests
import psycopg2
from psycopg2 import sql
import tempfile
import os
import mysql.connector
from dotenv import load_dotenv
import os
import logging
import boto3

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
db = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    dbname=os.getenv("DB_DATABASE"),
    port=os.getenv("DB_PORT")  # Add this line
)

cursor = db.cursor()

app = FastAPI()

# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

import boto3

s3 = boto3.client('s3')

bucket_name = os.getenv("BUCKET_NAME")
object_name = os.getenv("OBJECT_NAME")

s3.download_file(bucket_name, object_name, './models/best.pt')

# # Model URL and download logic
# url = "https://github.com/AbelBekele/ML-Microservice-Deployment/raw/main/model/best.pt"


def download_model():
    model_path = './models/best.pt'
    if not os.path.exists(model_path):
        try:
            pathlib.Path('./models').mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket_name, object_name, model_path)
            requests_logger.info(f"Model downloaded successfully to: {model_path}")
        except Exception as e:
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
        model_logger.error(f"Exception type: {type(e)}")
        model_logger.error(f"Exception args: {e.args}")
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

        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Results (
                id SERIAL PRIMARY KEY,
                image BYTEA,
                results TEXT
            )
        """)
        db.commit()

        # Insert image and results into the table
        cursor.execute("""
            INSERT INTO Results (image, results) VALUES (%s, %s)
        """, (psycopg2.Binary(image_bytes), counts_str))
        db.commit()

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