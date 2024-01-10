import pytest
import pinecone
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest import *
from db import *
from langchain.embeddings.openai import OpenAIEmbeddings
from bson.objectid import ObjectId

@pytest.fixture
def directory():
    directory = "pdf_resources/pdf_archive"
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
def collection_dict():
    collection_dict = {'_id':ObjectId('64d4125e2d9bf813089ca791'),'name':"Joe's Case"}
    return collection_dict

@pytest.fixture
def law_area():
    law_area = 'criminal_law'
    return law_area

@pytest.fixture
def index_name():
    index_name = "test4"
    return index_name

@pytest.fixture
def collection_name():
    collection_name = "Casandra's Case"
    return collection_name

@pytest.fixture
def index(index_name, embeddings, collection_name):
    index = get_existing_index(index_name, embeddings, collection_name)
    return index

@pytest.fixture
def file_name():
    file_name = '2017nbca10.pdf'
    #file_name = 'Khill-en.pdf'
    return file_name

def test_extract_summary(law_area, index, file_name):
    summary = extract_summary(law_area, index, file_name)
    print(summary)
    assert summary
    
#@pytest.mark.skip(reason="skiiiiiiip")
def test_process_pdfs(directory, embeddings, index_name, collection_dict, law_area):

    # Step 1: create user
    user_email = "gary.smith@gmail.com"
    first_name = "Gary"
    last_name = "Smith"
    user_id = insert_new_user(user_email, first_name, last_name)

     # Step 2: create collection for the user
    collection_name = "Rebecca's Case"
    collection_description = "Rebecca's troublesome history with drunk driving"
    collection_id = insert_new_collection(user_email, collection_name, collection_description)

    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment="northamerica-northeast1-gcp"
    )

    find = {'_id':collection_id}

    # Number of summaries before processing
    summary_count_before = len(collection_col.find_one(find)['case_summary_ids'])

    # Call process function
    process_pdfs(directory, embeddings, index_name, collection_name, collection_id, law_area)

    # Count files that were ingested
    _, _, files = next(os.walk(directory))
    file_count = len(files)

    # Number of summaries after processing
    summary_count_after = len(collection_col.find_one(find)['case_summary_ids'])

    difference = summary_count_after - summary_count_before

    assert difference == file_count