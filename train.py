import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
from google.cloud import storage

# Load CSV from Google Cloud Storage
def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

# Download the dataset
download_blob("my-ml-bucket", "dataset.csv", "dataset.csv")

# Read and process data
data = pd.read_csv("dataset.csv")
X = data.drop("price", axis=1)  # Features (e.g., size, bedrooms)
y = data["price"]  # Target (e.g., price)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

# Upload model to GCS
storage_client = storage.Client()
bucket = storage_client.bucket("my-ml-bucket")
blob = bucket.blob("model.pkl")
blob.upload_from_filename("model.pkl")
print("Model trained and uploaded to GCS!")
