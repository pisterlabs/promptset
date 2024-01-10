import json
import time
from datetime import datetime

from langchain.document_loaders import EverNoteLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_evernote(file_path):
    loader = EverNoteLoader(file_path, load_single_document=False)
    docs = loader.load()
    return docs

file_path = '<<PATH>>.enex'

doc = load_evernote(file_path)
print("Number of docs created from evernote doc is: " + str(len(doc)))
#print(doc)

chunk_size = 1000
chunk_overlap = 100
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunked_docs = text_splitter.split_documents(doc)
number_of_chunks = len(chunked_docs)

print("PDF Doc[" + file_path + "] was chunked into [" + str(number_of_chunks) + "] docs")

# Convert he list of Langhcain Document into List of Json strings.
# Also format the created/updated metadata to date ints so we can do metadata filtering in a vector db
chunk_docs_json = []
for doc in chunked_docs:
    # Assuming doc.metadata['created'] is a time.struct_time object. Convert to string
    created_str = time.strftime("%Y%m%dT%H%M%SZ", doc.metadata['created'])
    updated_str = time.strftime("%Y%m%dT%H%M%SZ", doc.metadata['updated'])

    # Parse the date string to a datetime object
    create_dt = datetime.strptime(created_str, "%Y%m%dT%H%M%SZ")
    update_dt = datetime.strptime(updated_str, "%Y%m%dT%H%M%SZ")

    # Convert the datetime object to a Unix timestamp with only the date component
    # This is done so we can store this is a metadata time column in a vector db

    doc.metadata['created'] = int(create_dt.timestamp())
    doc.metadata['updated'] = int(update_dt.timestamp())

    doc_json = doc.json()
    chunk_docs_json.append(doc_json)

# Convert the List of json strings into a single json string that can be return in the Flow File Result
single_json_doc = json.dumps(chunk_docs_json)

print(single_json_doc)