import os

import faiss
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.docsearch.faiss_qa import (
    embed_file,
    query_text_qa,
    retrieve_faiss_indexes_from_text,
)
from src.docsearch.pdf_parser import parse_pdf

load_dotenv()

PDF_TEST_DIR = "./data/pdf_samples/"

NUM_DIMENSIONS = 1536
EMBEDDINGS_MODEL = OpenAIEmbeddings(model="text-embedding-ada-002")
FAISS_INDEX = faiss.IndexFlatL2(NUM_DIMENSIONS)
CHAT_LLM = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_API_ENGINE"), temperature=0
)
# chunk up data to smaller documents
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=7_000, chunk_overlap=400)
TEXT = "What are these documents about?"


def test_pdf_query():
    # get all files in PDF_TEST_DIR
    pdf_files = [os.path.join(PDF_TEST_DIR, f) for f in os.listdir(PDF_TEST_DIR)][
        :3
    ]  # only take first 3 files to speed things up

    index_doc_store = {}
    counter = 0

    # loop through files and add them to index
    for uf in pdf_files:
        if not uf.endswith((".pdf")):
            continue

        embedded_texts, file_texts = embed_file(
            uf,
            EMBEDDINGS_MODEL,
            TEXT_SPLITTER,
            parse_pdf,
        )
        index_doc_store.update({i + counter: t for i, t in enumerate(file_texts)})
        counter += len(file_texts)
        FAISS_INDEX.add(embedded_texts)

    # check that the index_doc_store and faiss index is not empty
    assert len(index_doc_store) > 0
    assert FAISS_INDEX.ntotal > 0
    assert len(index_doc_store) == FAISS_INDEX.ntotal

    # retrieve faiss indexes from text based on query text
    faiss_idxs = retrieve_faiss_indexes_from_text(
        TEXT,
        FAISS_INDEX,
        EMBEDDINGS_MODEL,
    )

    assert len(faiss_idxs) > 0

    # query indexes to get result
    res = query_text_qa(
        TEXT,
        index_doc_store,
        CHAT_LLM,
        faiss_idxs,
    )

    assert len(res) > 0
    assert isinstance(res, str)

    print(res)
