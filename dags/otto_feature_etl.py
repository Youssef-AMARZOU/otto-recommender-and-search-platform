"""Airflow DAG: OTTO feature ETL (M1 scaffold).

Pipeline: extract raw sessions -> clean -> build user/item/session features ->
materialize into the Feast feature store so training and serving read identical
feature definitions.

Task callables are placeholders until milestone M1; running the DAG before then
fails loudly on the first task.
"""

from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

FEATURE_STORE_CONF = "configs/feature_store.yaml"


def extract_raw_sessions(**_):
    raise NotImplementedError("M1: load OTTO train.parquet sessions into the raw layer")


def clean_sessions(**_):
    raise NotImplementedError("M1: deduplicate, sort, and sessionize raw events")


def build_features(**_):
    raise NotImplementedError("M1: compute user/item/session features into the staging store")


def materialize_features(**_):
    raise NotImplementedError("M1: feast materialize-incremental with FEATURE_STORE_CONF")


with DAG(
    dag_id="otto_feature_etl",
    default_args=DEFAULT_ARGS,
    description="OTTO session logs -> Feast feature store",
    schedule_interval="@daily",
    start_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
    catchup=False,
    tags=["otto", "features"],
) as dag:
    extract = PythonOperator(task_id="extract_raw_sessions", python_callable=extract_raw_sessions)
    clean = PythonOperator(task_id="clean_sessions", python_callable=clean_sessions)
    build = PythonOperator(task_id="build_features", python_callable=build_features)
    materialize = PythonOperator(task_id="materialize_features", python_callable=materialize_features)

    extract >> clean >> build >> materialize
