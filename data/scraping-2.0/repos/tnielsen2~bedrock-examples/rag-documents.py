import os
import sys
from urllib.request import urlretrieve

import numpy as np
import psycopg2
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector

from utils import bedrock, print_ww

# docker run --name postgres -e POSTGRES_PASSWORD=bedrockworkshop! -p 5432:5432 -v ./init.sql:/docker-entrypoint-initdb.d/init.sql -d ankane/pgvector


# query = "What website can I visit for help with my taxes?"
# query = "When does a business have to report tips and what are the guidelines?"
# query = "Am I required to have an EIN to file taxes?"
# query = "What is the difference between a sole proprietorship and a partnership?"
# query = "Is it possible that I get sentenced to jail due to failure in filings?"
query = "What are the VPC architectures available for deployment within the AWS network architectural guidelines?"


def print_separator():
    print_ww(
        "***************************************************************************************\n"
        "***************************************************************************************\n"
        "\n"
        "\n"
    )


def bedrock_connection():
    return bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        region=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
    )


def init_bedrock_claude_client():
    return Bedrock(
        model_id="anthropic.claude-v1",
        client=bedrock_connection(),
        model_kwargs={"max_tokens_to_sample": 200},
    )


pg_params = {
    "host": "127.0.0.1",
    "database": os.environ.get("PGVECTOR_DATABASE", "postgres"),
    "user": os.environ.get("PGVECTOR_USER", "postgres"),
    "password": os.environ.get("PGVECTOR_PASSWORD", "bedrockworkshop!"),
}


def create_vector_datastore_connection_string(host, database, user, password):
    DRIVER = os.environ.get(
        "PGVECTOR_DRIVER",
        "psycopg2",
    )
    HOST = os.environ.get("PGVECTOR_HOST", host)
    PORT = os.environ.get("PGVECTOR_PORT", "5432")
    DATABASE = os.environ.get("PGVECTOR_DATABASE", database)
    USER = os.environ.get("PGVECTOR_USER", user)
    PASSWORD = os.environ.get("PGVECTOR_PASSWORD", password)

    CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver=DRIVER,
        host=HOST,
        port=PORT,
        database=DATABASE,
        user=USER,
        password=PASSWORD,
    )
    return CONNECTION_STRING


def fetch_irs_documents():
    os.makedirs("data", exist_ok=True)
    files = [
        "https://www.irs.gov/pub/irs-pdf/p1544.pdf",
        "https://www.irs.gov/pub/irs-pdf/p15.pdf",
        "https://www.irs.gov/pub/irs-pdf/p1212.pdf",
    ]
    for url in files:
        file_path = os.path.join("data", url.rpartition("/")[2])
        urlretrieve(url, file_path)


def load_vector_db(conn_string):
    module_path = ".."
    sys.path.append(os.path.abspath(module_path))
    bedrock_emb = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1", client=bedrock_connection()
    )

    fetch_irs_documents()

    loader = PyPDFDirectoryLoader("./data/")

    documents = loader.load()
    # Define text splitter object
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a tiny chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=100,
    )
    docs = text_splitter.split_documents(documents)

    avg_doc_length = lambda documents: sum(
        [len(doc.page_content) for doc in documents]
    ) // len(documents)
    avg_char_count_pre = avg_doc_length(documents)
    avg_char_count_post = avg_doc_length(docs)
    print(
        f"Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters."
    )
    print(
        f"After the split we have {len(docs)} documents more than the original {len(documents)}."
    )
    print(
        f"Average length among {len(docs)} documents (after split) is {avg_char_count_post} characters."
    )
    # Create a sample embedding using numpy
    sample_embedding = np.array(bedrock_emb.embed_query(docs[0].page_content))
    print("Sample embedding of a document chunk: ", sample_embedding)
    print("Size of the embedding: ", sample_embedding.shape)
    # call function that loads db and creates vectorstore
    load_and_search_embeddings(bedrock_emb, docs, conn_string)

    return bedrock_emb, docs


def load_and_search_embeddings(emb_obj, split_docs, vectorstore_connection_string):
    print("Loading documents into vectorstore...")
    db = PGVector.from_documents(
        embedding=emb_obj,
        documents=split_docs,
        collection_name="pdf_docs",
        connection_string=vectorstore_connection_string,
        pre_delete_collection=True,
    )
    # search db from datastore object
    print(f"Titan searching db from documents with query: {query}")
    docs_with_score = db.similarity_search_with_score(query)
    for doc, score in docs_with_score:
        print("-" * 80)
        print("Score: ", score)
        print(doc.page_content)
        print("-" * 80)


vector_db_connection_string = create_vector_datastore_connection_string(**pg_params)
bedrock_embeddings, qa_docs = load_vector_db(vector_db_connection_string)
print_separator()
print("Creating vectorstore from existing database...")
## Create vectorstore from existing database
tax_vectorstore = PGVector(
    collection_name="pdf_docs",
    connection_string=vector_db_connection_string,
    embedding_function=BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1", client=bedrock_connection()
    ),
)

claude_llm = init_bedrock_claude_client()

similar_documents = tax_vectorstore.similarity_search(query, k=3)
# our search query  # return 3 most relevant docs
print_separator()
print(f"Titan printing similar documents...{similar_documents}")

# Use claude to generate a prompt for the query
qa = RetrievalQA.from_chain_type(
    llm=claude_llm,
    chain_type="stuff",
    retriever=tax_vectorstore.as_retriever(),
)

qa_result = qa(query)
print_separator()
print_ww(f'Claude qa query standalone result: {qa_result["result"]}')

qa_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=claude_llm,
    chain_type="stuff",
    retriever=tax_vectorstore.as_retriever(),
    return_source_documents=True,
    # chain_type_kwargs={"prompt": PROMPT},
)

qa_source_result = qa_sources(query)
print_separator()
print_ww(f'Claude qa with sources answer: {qa_source_result["answer"]}')
try:
    print_ww(f'Claude qa with sources sources: {qa_source_result["source_documents"]}')
except KeyError:
    pass
