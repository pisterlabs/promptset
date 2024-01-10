import os
from functools import partial

import faiss
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.docsearch.faiss_qa import (
    embed_file,
    query_text_qa,
    retrieve_faiss_indexes_from_text,
)
from src.docsearch.ocr_parser import extract_text_from_img

# load environment variables
load_dotenv()

IMG_TEST_DIR = "./data/img_samples"

NUM_DIMENSIONS = 1536
EMBEDDINGS_MODEL = OpenAIEmbeddings(model="text-embedding-ada-002")
FAISS_INDEX = faiss.IndexFlatL2(NUM_DIMENSIONS)
CHAT_LLM = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_API_ENGINE"), temperature=0
)
# chunk up data to smaller documents
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=7_000, chunk_overlap=400)

ENDPOINT = os.getenv("FORM_RECOGNISER_ENDPOINT")
CREDENTIAL = AzureKeyCredential(os.getenv("FORM_RECOGNISER_KEY"))
DOC_ANALYSIS_CLIENT = DocumentAnalysisClient(ENDPOINT, CREDENTIAL)

TEXT = "Why did Nicole resign?"


def test_ocr_query():
    """Parses text from images and adds them to FAISS Index and index doc store,
    which has {index: text_chunk} for key:value.
    Then queries faiss index and returns relevant text for queries and finally
    returns the answer.
    """

    # get all files in IMG_TEST_DIR
    img_files = [os.path.join(IMG_TEST_DIR, f) for f in os.listdir(IMG_TEST_DIR)]

    # create a partial function to pass to add_files_to_index
    img_extract_fn = partial(
        extract_text_from_img,
        document_analysis_client=DOC_ANALYSIS_CLIENT,
    )

    # initialise index_doc_store; this is a dictionary that stores the index:document text
    index_doc_store = {}
    counter = 0

    # loop through files and add them to index
    for uf in img_files:
        if not uf.endswith((".jpg", ".png", ".jpeg")):
            continue

        embedded_texts, file_texts = embed_file(
            uf,
            EMBEDDINGS_MODEL,
            TEXT_SPLITTER,
            img_extract_fn,
        )
        index_doc_store.update({i + counter: t for i, t in enumerate(file_texts)})
        counter += len(file_texts)
        FAISS_INDEX.add(embedded_texts)

    # check that the index_doc_store and faiss index is not empty
    assert len(index_doc_store) > 0
    assert FAISS_INDEX.ntotal > 0
    assert len(index_doc_store) == FAISS_INDEX.ntotal

    faiss_idxs = retrieve_faiss_indexes_from_text(
        TEXT,
        FAISS_INDEX,
        EMBEDDINGS_MODEL,
    )

    assert len(faiss_idxs) > 0

    res = query_text_qa(
        TEXT,
        index_doc_store,
        CHAT_LLM,
        faiss_idxs,
    )

    assert len(res) > 0
    assert isinstance(res, str)

    print(res)
