import json
import logging

from logging import config

from dotenv import load_dotenv
from langchain.vectorstores import Weaviate

from log_config import logging_config

load_dotenv()  # take environment variables from .env.
config.dictConfig(logging_config)  # Load the logging configuration

from langchainmod import load_content_vacs
from weaviatedb import WeaviateClient

WEAVIATE_CLASS = "RijksoverheidVac"

run_logging = logging.getLogger("runner")  # initialize the main logger


def load_weaviate_schema(client: WeaviateClient, schema_path: str) -> None:
    run_logging.info(f"(Re)Load the Weaviate schema class: {WEAVIATE_CLASS}")
    if client.does_class_exist(WEAVIATE_CLASS):
        client.delete_class(WEAVIATE_CLASS)
        run_logging.info("Removed the existing Weaviate schema.")

    client.create_classes(path_to_schema=schema_path)
    run_logging.info("New schema loaded.")


def load_content(client: WeaviateClient) -> None:
    vector_store = Weaviate(
        client=client.client,
        index_name=WEAVIATE_CLASS,
        text_key="question",
        attributes=["dataurl"]
    )

    load_content_vacs(vector_store=vector_store, rows=20)


def q_and_a(client: WeaviateClient, question: str):
    ask = {
        "question": question,
        "properties": ["antwoord"]
    }

    return (
        client.client.query
        .get(WEAVIATE_CLASS, ["question", "antwoord",
                              "_additional {answer {hasAnswer certainty property result startPosition endPosition} }"])
        .with_ask(content=ask)
        .with_limit(1)
        .do()
    )


def query(client: WeaviateClient, query_text: str):
    near_text = {"concepts": [query_text]}

    return (
        client.client.query
        .get(WEAVIATE_CLASS, ["question","antwoord"])
        .with_near_text(near_text)
        .with_limit(5)
        .do()
    )


def print_result(result):
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    weaviate_client = WeaviateClient()

    load_weaviate_schema(client=weaviate_client, schema_path="./config_files/rovac_weaviate_schema_local.json")
    load_content(client=weaviate_client)

    print_result(weaviate_client.inspect())

    the_query = "hoeveel mag ik drinken als ik moet rijden?"
    print_result(query(client=weaviate_client, query_text=the_query))

    answer_response = q_and_a(client=weaviate_client, question=the_query)
    print_result(answer_response)
