from airflow import DAG
from airflow.providers.google.cloud.operators.vertex_ai import VertexAICustomTrainingJobOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 3, 1),
    "retries": 1,
}

with DAG("train_model_dag", default_args=default_args, schedule_interval=None) as dag:
    train_task = VertexAICustomTrainingJobOperator(
        task_id="train_model",
        project_id="my-ml-project",  # Replace with your GCP project ID
        region="us-central1",
        staging_bucket="gs://my-ml-bucket",
        script_path="train.py",
        container_uri="gcr.io/cloud-aiplatform/training/sklearn-cpu.0-23:latest",
    )
