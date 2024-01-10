import os
import time
from datetime import datetime

from airflow import DAG
from airflow.decorators import setup, task, teardown
from airflow.providers.cohere.operators.embedding import CohereEmbeddingOperator
from airflow.providers.pinecone.hooks.pinecone import PineconeHook
from airflow.providers.pinecone.operators.pinecone import PineconeIngestOperator

index_name = os.getenv("INDEX_NAME", "example-pinecone-index")
namespace = os.getenv("NAMESPACE", "example-pinecone-index")
data = [
    "Alice Ann Munro is a Canadian short story writer who won the Nobel Prize in Literature in 2013. Munro's work has been described as revolutionizing the architecture of short stories, especially in its tendency to move forward and backward in time."
]

with DAG(
    "example_pinecone_cohere",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
    # [START howto_operator_pinecone_ingest]

    @setup
    @task
    def create_index():
        hook = PineconeHook()
        hook.create_index(index_name=index_name, dimension=768)
        time.sleep(60)

    embed_task = CohereEmbeddingOperator(
        task_id="embed_task",
        input_text=data,
    )

    perform_ingestion = PineconeIngestOperator(
        task_id="perform_ingestion",
        index_name=index_name,
        input_vectors=[
            ("id1", embed_task.output),
        ],
        namespace=namespace,
        batch_size=1,
    )

    @teardown
    @task
    def delete_index():
        hook = PineconeHook()
        hook.delete_index(index_name=index_name)

    create_index() >> embed_task >> perform_ingestion >> delete_index()
