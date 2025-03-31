import os
import gdown
import pandas as pd

# Google Drive File ID (Replace with your actual File ID)
FILE_ID = "1hE0fFEyzdJUyJ1pMdqp8yIdGXyYMFnhr"
DATA_DIR = "../data"
FILE_PATH = os.path.join(DATA_DIR, "creditcard.csv")

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Check if file exists, if not, download from Google Drive
if not os.path.exists(FILE_PATH):
    print("Downloading dataset from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, FILE_PATH, quiet=False)
else:
    print("Dataset already exists. Skipping download.")

# Load dataset
df = pd.read_csv(FILE_PATH)
print("Dataset Loaded Successfully!")
print(df.head())  # Print first 5 rows to confirm
