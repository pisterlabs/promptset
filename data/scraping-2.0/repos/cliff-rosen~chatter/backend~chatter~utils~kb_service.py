from db import db
from utils.utils import num_tokens_from_string
from utils.logging import logging
import utils.openai_wrappers as model
import utils.pinecone_wrappers as vdb
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

COMPLETION_MODEL = 'text-davinci-003'
TEMPERATURE = 0.0

logger = logging.getLogger()

'''
add_document(domain_id, uri, title, text, blob)
update_document(doc_id, uri, title, text, blob)
del_document(doc_id)
get_document(doc_id)
get_chunk(doc_chunk_id)
get_parent_document(doc_chunk_id)
get_chunks_from-query(domain_id, query)

chunks dict: 
    {
        ID:  {
            "id": ID as int,
            "score": score as float,
            "metadata": {
                "doc_chunk_id": 44743.0,
                "doc_id": 20657.0,
                "domain_id": 27.0                
            },
            "uri": uri,
            "text": text,
            "used": isUsed
        }
    }

    {
        "27": {
            "id": 27,
            "score": 0.737494111,
            "metadata": {
                "doc_chunk_id": 27.0,
                "doc_id": 15.0,
                "domain_id": 1.0
            },
            "uri": "Changes in Drug Level Laboratory Results _ DoseMe Help Center.pdf",
            "text": "different, DoseMeRx will tend to prefer the one most like the population model (as this is more \ncommon in the population). Therefore, it may recommend a different dose than what would be \ncustomary for a patient if only the most recent result was considered.\nHere are two approaches to consider when this is encountered:\nIf the accuracy of the outlier drug level is questionable:\n\u0000. Consider obtaining another level if possible to validate the accuracy of the most recent \nlevel.\n\u0000. If you cannot obtain a level, exclude the last level and DoseMeRx will calculate the dose \nbased on the prior existing levels.\nIf the most recent drug level value is considered to be correct:\n9/14/23, 4:09 PM Changes in Drug Level Laboratory Results | DoseMe Help Center\nhttps://help.doseme-rx.com/en/articles/3353676-changes-in-drug-level-laboratory-results 2/2doseme-rx.com\n\u0000. Exclude earlier drug levels (if the last result is considered correct and you think a change \nhas taken place).",
            "used": true
        }    
    }
'''

def _embed_and_add_document_chunk(doc_id, chunk_text):
    emb = model.get_embedding(chunk_text)
    doc_chunk_id = db.insert_document_chunk(doc_id, chunk_text, emb)
    return (emb, doc_chunk_id)

def _make_chunks_from_text(text):
    chunks_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap  = 0,
            length_function = len,
        )
    chunks = chunks_splitter.split_text(text)
    print('Chunks produced:', len(chunks))
    return chunks


# mutate chunks by adding {"uri": uri, "text", text} to each value dict
# chunks is dict where
#   key is chunk_id, and value is obj with score, text
def _set_chunk_text_from_ids(chunks):
    ids = list(chunks.keys())
    rows = db.get_document_chunks_from_ids(ids)
    for row in rows:
        doc_chunk_id = row["doc_chunk_id"]
        chunk_text = row["chunk_text"]
        doc_uri = row["doc_uri"]
        print(f"id: {doc_chunk_id}, text: {chunk_text[:20]}")
        chunks[str(doc_chunk_id)]["uri"] = doc_uri
        chunks[str(doc_chunk_id)]["text"] = chunk_text


# mutate chunks by adding {"uri": uri, "text", text} to each value dict
# chunks is dict where
#   key is chunk_id, and value is obj with score, text
def _set_chunk_text_from_ids(chunks):
    ids = list(chunks.keys())
    rows = db.get_document_chunks_from_ids(ids)
    for row in rows:
        doc_chunk_id = row["doc_chunk_id"]
        chunk_text = row["chunk_text"]
        doc_uri = row["doc_uri"]
        print(f"id: {doc_chunk_id}, text: {chunk_text[:20]}")
        chunks[str(doc_chunk_id)]["uri"] = doc_uri
        chunks[str(doc_chunk_id)]["text"] = chunk_text


def add_document(domain_id, uri, title, text, blob):
    doc_id = db.insert_document(domain_id, uri, title, text, text)
    chunks = _make_chunks_from_text(text)
    for chunk in chunks:
        (emb, doc_chunk_id) = _embed_and_add_document_chunk(doc_id, chunk)
        vdb.upsert_index(doc_id, doc_chunk_id, emb, domain_id)
        print('added chunk ', doc_chunk_id)
    return doc_id

def delete_document(doc_id):
    vdb.delete(doc_id, {})
    db.delete_document(doc_id)

def delete_documents(doc_ids):
    for doc_id in doc_ids:
        vdb.delete_all_for_doc_id(doc_id)
        db.delete_document(doc_id)

def get_chunks_from_query(domain_id, user_message):
    chunks = {}

    print("getting query embedding")
    query_embedding = model.get_embedding(user_message)

    print("getting chunks ids")
    chunks = vdb.get_matching_chunks(domain_id, query_embedding)
    if not chunks:
        raise Exception('No chunks found - check index')

    print("getting chunk text from ids")
    _set_chunk_text_from_ids(chunks)

    #logger.info(chunks)
    return chunks

