"""
This runner uses langchain to index content from the Dutch Rijksoverheid about Vraag Antwoord Combinaties. The project
uses langchain to send the data to OpenSearch as well as Weaviate. Results can be compared, but for me the needed
code for both solutions is more important at the moment.
"""
import logging
import os
from logging import config

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Weaviate, VectorStore, OpenSearchVectorSearch
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from langchainmod import VraagXMLLoader, AntwoordXMLLoader, load_content_vacs
from log_config import logging_config
from search import OpenSearchClient
from weaviatedb import WeaviateClient

WEAVIATE_CLASS = "RijksoverheidVac"
OPENSEARCH_INDEX = "rijksoverheid-vac"

load_dotenv()  # take environment variables from .env.
config.dictConfig(logging_config)  # Load the logging configuration
run_logging = logging.getLogger("runner")  # initialize the main logger


def load_weaviate_schema(weaviate_client: WeaviateClient) -> None:
    run_logging.info(f"(Re)Load the Weaviate schema class: {WEAVIATE_CLASS}")
    if weaviate_client.does_class_exist(WEAVIATE_CLASS):
        weaviate_client.delete_class(WEAVIATE_CLASS)
        run_logging.info("Removed the existing Weaviate schema.")

    weaviate_client.create_classes("./config_files/rovac_weaviate_schema.json")
    run_logging.info("New schema for langchain rijksoverheid vac loaded.")


def run_weaviate(query: str = "enter your query", do_load_content: bool = False) -> None:
    weaviate_client = WeaviateClient()
    vector_store = Weaviate(
        client=weaviate_client.client,
        index_name=WEAVIATE_CLASS,
        text_key="question",
        attributes=["dataurl"]
    )

    if do_load_content:
        load_weaviate_schema(weaviate_client=weaviate_client)
        load_content_vacs(vector_store=vector_store, rows=200)

    index = VectorStoreIndexWrapper(vectorstore=vector_store)
    docs = index.vectorstore.similarity_search(
        query=query,
        search_distance=0.6,
        additional=["certainty"],
        k=10)

    print(f"\nResults from: Weaviate")
    for doc in docs:
        print(f"{doc.metadata['_additional']['certainty']} - {doc.page_content} -  - {doc.metadata['dataurl']}")


def use_weaviate_for_qa(query: str = "enter your query"):
    vector_store = Weaviate(
        client=WeaviateClient().client,
        index_name=WEAVIATE_CLASS,
        text_key="antwoord",
        attributes=["dataurl"]
    )

    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the 
    answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer in Dutch:"""
    custom_prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": custom_prompt}
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=os.getenv('OPEN_AI_API_KEY')),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs=chain_type_kwargs
    )
    answer = qa.run(query)

    print(answer)


def run_opensearch(query: str = "enter your query", do_load_content: bool = False) -> None:
    auth = (os.getenv('OS_USERNAME'), os.getenv('OS_PASSWORD'))
    opensearch_client = OpenSearchClient()

    vector_store = OpenSearchVectorSearch(
        index_name=OPENSEARCH_INDEX,
        embedding_function=OpenAIEmbeddings(openai_api_key=os.getenv('OPEN_AI_API_KEY')),
        opensearch_url="https://localhost:9200",
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        http_auth=auth
    )

    if do_load_content:
        opensearch_client.delete_index(OPENSEARCH_INDEX)
        load_content_vacs(vector_store=vector_store, rows=200)

    docs = vector_store.similarity_search_with_score(query=query)
    print(f"\nResults from: OpenSearch")
    for doc, _score in docs:
        print(f"{_score} - {doc.page_content}")


if __name__ == '__main__':
    run_logging.info("Starting the script Langchain Rijksoverheid Vraag Antwoord Combinaties")

    query_str_11 = "Moet ik mijn hond laten chippen?"
    query_str_12 = "Waar kan ik mijn hond laten chippen?"
    query_str_13 = "Kan iemand anders dan de dierenarts mijn hond chippen?"
    query_str_14 = "Waar moet ik naar zoeken om mijn hond te chippen?"

    query_str_21 = "Kan ik met iemand praten over dementie?"
    query_str_22 = "Bij welke organisaties kan ik terecht om te praten over dementie?"
    query_str_23 = "Heeft u contact gegevens van organisaties waar ik terecht kan om te praten over dementie?"

    query_str_31 = "Hoe lang duurt het om een rijbewijs aan te vragen?"
    query_str_32 = "How long do I have to wait for my drivers license?"
    query_str_33 = "How long do I have to wait to get my drivers license?"
    query_str_34 = "hoeveel mag ik drinken als ik moet rijden?"

    query_str = query_str_21

    run_weaviate(query=query_str,
                 do_load_content=True)

    # run_opensearch(query=query_str,
    #                do_load_content=False)

    # use_weaviate_for_qa(query=query_str)