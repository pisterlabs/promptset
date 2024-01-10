import pytest
import pinecone
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest import *
from db import *
from bson.objectid import ObjectId
from langchain.embeddings.openai import OpenAIEmbeddings
from evaluate_cases import evaluate_relevancy_for_summaries_in_collection

@pytest.fixture
def directory():
    directory = "pdf_resources/std_test"
    return directory
    
@pytest.fixture
def embeddings():
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    return embeddings

@pytest.fixture
def index_name():
    index_name = "test4"
    return index_name

@pytest.fixture
def law_area():
    law_area = 'criminal_law'
    return law_area

@pytest.fixture
def fact_sheet_file_path():
    fact_sheet_file_path = 'fact_sheet/std_fact_pattern.pdf'
    return fact_sheet_file_path

@pytest.fixture
def fact_sheet_file_name():
    fact_sheet_file_name = 'std_fact_pattern.pdf'
    return fact_sheet_file_name

def test_evaluate_relevancy_for_summaries_in_collection(directory, embeddings, index_name, law_area, fact_sheet_file_path, fact_sheet_file_name):
    user_email = "gary.smith@gmail.com"
    first_name = "Gary"
    last_name = "Smith"
    
    # Step 1: create a new user
    user_id = insert_new_user(user_email, first_name, last_name)

    # Step 2: create collection for the user
    collection_name = "Frank's Case"
    collection_description = "Frank's troublesome history with passing on STD's"
    collection_id = insert_new_collection(user_email, collection_name, collection_description)

    # Step 3: insert fact sheet
    fact_sheet_string = pdf_to_string(fact_sheet_file_path)
    fact_sheet_id = insert_new_fact_sheet(collection_id, fact_sheet_file_name, fact_sheet_string)

    # Step 4: summarize PDFs
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment="northamerica-northeast1-gcp"
    )

    # Call process function
    process_pdfs(directory, embeddings, index_name, collection_name, collection_id, law_area, api_mode=False)

    # Step 5: evaluate relevancies
    relevancy = evaluate_relevancy_for_summaries_in_collection(collection_id, fact_sheet_id)
    print(relevancy)
    assert len(relevancy) == len(os.listdir(directory))