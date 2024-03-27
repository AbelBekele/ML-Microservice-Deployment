import os
import zipfile
from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi

# Instantiate the Kaggle API client
api = KaggleApi()

api.authenticate()

# Define the directory path where you want to download the dataset
download_dir = "content"

# Download the dataset into the specified directory
api.dataset_download_files(dataset="gauravduttakiit/clickthrough-rate-prediction", path=download_dir, unzip=True)
