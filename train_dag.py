from airflow import DAG
from airflow.providers.google.cloud.operators.vertex_ai import VertexAICustomTrainingJobOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "train_dag",
    start_date=datetime(2024, 3, 8),
    schedule_interval="@daily",
    catchup=False,
    default_args=default_args,
) as dag:
    train_task = VertexAICustomTrainingJobOperator(
        task_id="run_pipeline",
        project_id="gen-lang-client-0736387453",
        region="us-central1",
        staging_bucket="gs://lawblowbucket",
        script_path="train.py",
        container_uri="gcr.io/cloud-aiplatform/training/sklearn-cpu.0-23:latest",
    )
