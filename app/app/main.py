from fastapi import FastAPI, UploadFile, File
from typing import List
import cv2
import numpy as np
import torch
import pandas as pd
import pathlib

app = FastAPI()

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model = torch.hub.load('ultralytics/yolov5', 'custom', './yolov5/runs/train/exp4/weights/best.pt',  _verbose=False)  # custom trained model
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