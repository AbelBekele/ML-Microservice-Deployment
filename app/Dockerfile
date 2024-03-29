FROM ultralytics/yolov5:latest

# Copy model and application files
WORKDIR /app
COPY . /app

# Install additional dependencies (if needed)
RUN pip install -r requirements.txt

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
