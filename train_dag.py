#!/usr/bin/env python3
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from google.cloud import storage
from airflow import DAG
from airflow.operators.python import PythonOperator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GCS bucket details
BUCKET_NAME = "lawblowbucket"
DATA_DIR = "data/"
PROCESSED_DIR = "processed_data/"
MODEL_DIR = "models/"

# Helper functions for GCS
def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logger.info(f"Downloaded {source_blob_name} to {destination_file_name}")

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    logger.info(f"Uploaded {source_file_name} to {destination_blob_name}")

# ETL function
def etl():
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = [blob for blob in bucket.list_blobs(prefix=DATA_DIR) if blob.name.endswith('.csv')]
        excluded_files = {f"{DATA_DIR}books_data.csv", f"{DATA_DIR}Books_rating.csv", f"{DATA_DIR}Liquor_Sales.csv"}
        csv_blobs = [blob for blob in blobs if blob.name not in excluded_files]
        
        if not csv_blobs:
            raise ValueError("No valid CSV files found in GCS data directory")
        
        logger.info(f"Found {len(csv_blobs)} CSV files: {[blob.name for blob in csv_blobs]}")
        dataframes = [pd.read_csv(f"gs://{BUCKET_NAME}/{blob.name}") for blob in csv_blobs]
        sales_data = pd.concat(dataframes, ignore_index=True)
        
        output_file = "/tmp/cleaned_data.csv"
        sales_data.to_csv(output_file, index=False)
        upload_blob(BUCKET_NAME, output_file, f"{PROCESSED_DIR}cleaned_data.csv")
        logger.info(f"ETL completed. Saved to {PROCESSED_DIR}cleaned_data.csv")
    except Exception as e:
        logger.error(f"ETL failed: {str(e)}")
        raise

# Feature engineering function
def feature_engineering():
    try:
        input_file = "/tmp/cleaned_data.csv"
        download_blob(BUCKET_NAME, f"{PROCESSED_DIR}cleaned_data.csv", input_file)
        sales_data = pd.read_csv(input_file)
        logger.info(f"Loaded {len(sales_data)} rows from {input_file}")
        
        sales_data["Order Date"] = pd.to_datetime(sales_data["Order Date"], errors='coerce')
        sales_data.dropna(subset=["Order Date"], inplace=True)
        sales_data["Month"] = sales_data["Order Date"].dt.month
        sales_data["Hour"] = sales_data["Order Date"].dt.hour
        
        output_file = "/tmp/feature_engineered_data.csv"
        sales_data.to_csv(output_file, index=False)
        upload_blob(BUCKET_NAME, output_file, f"{PROCESSED_DIR}feature_engineered_data.csv")
        logger.info(f"Feature engineering completed. Saved to {PROCESSED_DIR}feature_engineered_data.csv")
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        raise

# Model training function
def train_model():
    try:
        input_file = "/tmp/feature_engineered_data.csv"
        download_blob(BUCKET_NAME, f"{PROCESSED_DIR}feature_engineered_data.csv", input_file)
        data = pd.read_csv(input_file)
        logger.info(f"Loaded {len(data)} rows for training")
        
        X = data[["Month", "Hour", "Quantity Ordered"]]
        y = data["Price Each"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        model_file = "/tmp/sales_model.pkl"
        joblib.dump(model, model_file)
        upload_blob(BUCKET_NAME, model_file, f"{MODEL_DIR}sales_model.pkl")
        logger.info("Model training completed")
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

# Validation function
def validate_results():
    try:
        cleaned_file = "/tmp/cleaned_data.csv"
        download_blob(BUCKET_NAME, f"{PROCESSED_DIR}cleaned_data.csv", cleaned_file)
        cleaned_data = pd.read_csv(cleaned_file)
        logger.info(f"Validated cleaned_data.csv: {len(cleaned_data)} rows")

        feature_file = "/tmp/feature_engineered_data.csv"
        download_blob(BUCKET_NAME, f"{PROCESSED_DIR}feature_engineered_data.csv", feature_file)
        feature_data = pd.read_csv(feature_file)
        logger.info(f"Validated feature_engineered_data.csv: {len(feature_data)} rows")

        model_file = "/tmp/sales_model.pkl"
        download_blob(BUCKET_NAME, f"{MODEL_DIR}sales_model.pkl", model_file)
        model = joblib.load(model_file)
        sample_input = feature_data[["Month", "Hour", "Quantity Ordered"]].iloc[0:1]
        sample_prediction = model.predict(sample_input)[0]
        logger.info(f"Validated model: Sample prediction = {sample_prediction:.2f}")
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise

# Define the DAG
with DAG(
    "Main",
    start_date=datetime(2024, 3, 8),
    schedule_interval="@daily",
    catchup=False,
    default_args={
        "owner": "airflow",
        "retries": 3,
        "retry_delay": timedelta(minutes=5),
    },
) as dag:
    etl_task = PythonOperator(task_id="extract_transform_load", python_callable=etl)
    feature_engineering_task = PythonOperator(task_id="feature_engineering", python_callable=feature_engineering)
    train_model_task = PythonOperator(task_id="train_model", python_callable=train_model)
    validate_results_task = PythonOperator(task_id="validate_results", python_callable=validate_results)

    etl_task >> feature_engineering_task >> train_model_task >> validate_results_task
