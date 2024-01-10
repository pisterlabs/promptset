import json
from pathlib import Path

import pendulum
from airflow.decorators import dag, setup, task, teardown
from airflow.providers.cohere.operators.embedding import CohereEmbeddingOperator
from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
from airflow.providers.weaviate.operators.weaviate import WeaviateIngestOperator


@dag(
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=["example", "weaviate", "cohere"],
)
def example_cohere_weaviate():
    """
    Example DAG which creates embeddings using CohereEmbeddingOperator and the uses WeaviateIngestOperator to insert embeddings to Weaviate .
    """

    @setup
    @task
    def create_weaviate_class():
        """
        Example task to create class without any Vectorizer. You're expected to provide custom vectors for your data.
        """
        weaviate_hook = WeaviateHook()
        # Class definition object. Weaviate's autoschema feature will infer properties when importing.
        class_obj = {
            "class": "Weaviate_example_class_test",
            "vectorizer": "none",
        }
        weaviate_hook.create_class(class_obj)

    @setup
    @task
    def get_data_to_embed():
        data = json.load(Path("./dags/jeopardy_data_without_vectors.json").open())
        return [[item["Question"]] for item in data]

    data_to_embed = get_data_to_embed()
    embed_data = CohereEmbeddingOperator.partial(
        task_id="embedding_using_xcom_data",
    ).expand(input_text=data_to_embed["return_value"])

    @task
    def update_vector_data_in_json(**kwargs):
        ti = kwargs["ti"]
        data = json.load(Path("./dags/jeopardy_data_without_vectors.json").open())
        embedded_data = ti.xcom_pull(task_ids="embedding_using_xcom_data", key="return_value")
        for i, vector in enumerate(embedded_data):
            data[i]["Vector"] = vector[0]
        return data

    update_vector_data_in_json = update_vector_data_in_json()

    perform_ingestion = WeaviateIngestOperator(
        task_id="perform_ingestion",
        conn_id="weaviate_default",
        class_name="Weaviate_example_class_test",
        input_json=update_vector_data_in_json["return_value"],
    )

    embed_query = CohereEmbeddingOperator(
        task_id="embed_query",
        input_text=["biology"],
    )

    @teardown
    @task
    def delete_weaviate_class():
        """
        Example task to delete a weaviate class
        """
        weaviate_hook = WeaviateHook()
        # Class definition object. Weaviate's autoschema feature will infer properties when importing.

        weaviate_hook.delete_classes(["Weaviate_example_class_test"])

    (
        create_weaviate_class()
        >> embed_data
        >> update_vector_data_in_json
        >> perform_ingestion
        >> embed_query
        >> delete_weaviate_class()
    )


example_cohere_weaviate()
