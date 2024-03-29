from fastapi import FastAPI, UploadFile, File
from typing import List
import cv2
import numpy as np
import torch
import pandas as pd
import pathlib
import requests

app = FastAPI()

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# URL of the model
url = "https://github.com/AbelBekele/ML-Microservice-Deployment/raw/main/model/best.pt"

# Send HTTP request to the URL
response = requests.get(url, allow_redirects=True)

# Ensure the models directory exists
pathlib.Path('./models').mkdir(parents=True, exist_ok=True)

# Write the content of the response to a file in the models directory
with open('./models/best.pt', 'wb') as file:
    file.write(response.content)

# Now you can load the model from the models directory
model = torch.hub.load('ultralytics/yolov5', 'custom', './models/best.pt', _verbose=False)  # custom trained model
model.conf = 0.7  # NMS confidence threshold


class ImageType(UploadFile):
    data: bytes
    
@app.post("/predict")
async def predict(image: UploadFile):
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

# Run the FastAPI application locally
if __name__ == "__main__":
 import uvicorn
 uvicorn.run(app, host="0.0.0.0", port=8000)