from datetime import datetime

from airflow import DAG
from airflow.providers.cohere.operators.embedding import CohereEmbeddingOperator

with DAG("example_cohere_embedding", schedule=None, start_date=datetime(2023, 1, 1), catchup=False) as dag:
    # [START howto_operator_cohere_embedding]
    texts = [
        "On Kernel-Target Alignment. We describe a family of global optimization procedures",
        " that automatically decompose optimization problems into smaller loosely coupled",
        " problems, then combine the solutions of these with message passing algorithms.",
    ]

    def get_text():
        return texts

    CohereEmbeddingOperator(input_text=texts, task_id="embedding_via_text")
    CohereEmbeddingOperator(input_text=texts[0], task_id="embedding_via_task")
    # [END howto_operator_cohere_embedding]

if __name__ == "__main__":
    dag.test()
