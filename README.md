
# FastAPI Object Detection Microservice Deployment

This repository contains a microservice incorporating pre-trained machine learning models for object detection using FastAPI. The deployment process is streamlined with a comprehensive CI/CD pipeline, ensuring efficient management and reliable performance.

### 1. Setup Process

#### Prerequisites:

-   Python 3.6 or later
-   pip (Python package manager)
-   Docker

#### Installation:

1.  Create a virtual environment (recommended):
    
    `python -m venv env
    source env/bin/activate  # Linux/macOS
    env\Scripts\activate.bat  # Windows` 
    
2.  Install dependencies:
    
    `pip install -r requirements.txt` 
    
3.  Download the pre-trained model:
    -   From AWS S3 storage:
        
        `aws s3 cp s3://automation-challenge/models/best.pt models/best.pt` 
        
    -   OR from GitHub:
        
        `curl -L https://github.com/AbelBekele/ML-Microservice-Deployment/raw/main/model/best.pt -o models/best.pt` 
        
4.  Create a `.env` file in the project root directory:
       
    - DB_HOST=aws_rds_database_host
    - DB_USER=aws_rds_database_user
    - DB_PASSWORD=aws_rds_database_password
    - DB_DATABASE=aws_rds_database_name 
    

### 2. API Usage

The API provides two endpoints:

-   `/predict` (POST): Accepts an image file as input and returns detected objects and their counts.
-   `/health` (GET): Returns a simple health check response indicating if the service is running.

Request Body for `/predict`:

`{
  "image": (image file)
}` 

Response for `/predict` and `/health`:

`{
  "results": "object1 count, object2 count, ..."
}` 

Example Usage (using curl):

- `curl -X POST http://localhost:8000/predict -F "image=@image.jpg"`
- `curl http://localhost:8000/health` 

### 3. Using Docker for Containerization

Docker image creation utilizes the base image `ultralytics/yolov5:latest`. Python dependencies are installed via pip based on `requirements.txt`, with port 8000 exposed for external communication.

### 4. task-definition.json
Defines ECS task configuration for running Docker containers, specifying image, ports, memory, and CPU settings.

### 5. buildspec.yml
Specifies build steps for AWS CodeBuild, including logging in to ECR, building Docker image, and pushing to ECR.

### 6. GitHub Action
Automates testing and deployment process on push to main branch, including testing code and deploying Docker image to ECS.

### 7. AWS EC2 Permissions Checker Script
A script (check_permissions.py) provided to analyze IAM role permissions within an AWS EC2 instance, ensuring proper configurations and identifying potential issues.

## Contributions

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the [MIT License]