import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import re
import os
import sys
#import sys
#sys.path.append('.\..')
from db import db
import local_secrets as secrets

"""
embedding length: 1536

Retrieve all documents for domain
For each document
    break into chunks
    for each chunk
        get embedding
        insert chunk with embedding into document_chunk table
"""

MIN_CHUNK_LENGTH = 20
MAX_CHUNK_LENGTH = 1500

OPENAI_API_KEY = secrets.OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

def get_openai_embedding(text):
    embedding_model = "text-embedding-ada-002"
    return get_embedding(
        text,
        engine="text-embedding-ada-002"
    )

def get_all_docs_from_domain(conn, domain_id):
    return db.get_all_docs_from_domain(conn, domain_id)

def get_docs_from_ids(conn, ids):
    return db.get_docs_from_ids(conn, ids)

def get_chunks_from_text(text, maker_type):
    if maker_type == "MAKER_2":
        return get_chunks_from_text_maker_2(text)

    if maker_type == "CHAR":
        print('chunking with CharacterTextSplitter')
        chunks_splitter = CharacterTextSplitter(        
            #separator = "\n\n",
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
        )
    else:
        print('chunking with RecursiveCharacterTextSplitter')
        #text = re.sub('\s+', ' ', text)
        chunks_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap  = 0,
            length_function = len,
        )
    chunks = chunks_splitter.split_text(text)
    return chunks

# create fragments, which are chunks delimited by \n\n
# chunks are fragments concatenated until a fragment is min 20 words
def get_chunks_from_text_maker_2(text):
    print("chunk maker 2")
    chunks = []
    fragments = []

    # clean input
    text = text.encode(encoding='ASCII',errors='ignore').decode()
    text.strip()
    #text = re.sub('\s{3,}', '\n\n', text)    

    # build array of fragments by nn
    fragments = text.split('\n\n')

    # add array elements until reaching an element with at least 20 words
    cur_chunk = ""
    for i, fragment in enumerate(fragments):
        cur_chunk = cur_chunk + '\n' + fragment
        if len(cur_chunk) > 1 and (len(fragment.split()) >= 20 or i + 1 == len(fragments)):
            cur_chunk = cur_chunk.strip()
            if len(cur_chunk) > MIN_CHUNK_LENGTH:
                chunks.append(cur_chunk)
            cur_chunk = ""

    return chunks

# runtime settings
#chunk_maker = "MAKER_2"
#chunk_maker = "CHAR"
chunk_maker = "MAKER_1"
domain_id = 1
#doc_ids = None
doc_ids = [53, 54, 55, 56, 57]

def run():
    # init
    conn = db.get_connection()

    # one to one creation of chunks with embeddings
    # FIX ME: should be upsertChunk() and not insertChunk()
    if not doc_ids:
        print("Retrieve documents for domain", domain_id)
        rows = get_all_docs_from_domain(conn, domain_id)
    else:
        print("Retrieving documents: ", doc_ids)
        rows = get_docs_from_ids(conn, doc_ids)

    print("Retrieved: ", len(rows))

    for doc_id, _domain_id, uri, doc_title, doc_text in rows:
        print("****************************")
        chunks = get_chunks_from_text(doc_text, chunk_maker)
        print(uri, len(chunks))
        for chunk in chunks:
            print(doc_id, chunk[:50])
            print("----------------------")
            embedding = get_openai_embedding(chunk[:MAX_CHUNK_LENGTH])
            db.insert_document_chunk(conn, doc_id, chunk, embedding)

    # cleanup
    db.close_connection(conn)

def write_to_file(text):
    directory = 'chatter\data_processor\outputs'
    dest = 'chunks.txt'
    with open(os.path.join(directory, dest), 'a') as new_file:
        new_file.write(text)

def test_chunker():
    print("TEST: Retrieve documents for domain", domain_id)
    conn = db.get_connection()
    rows = get_all_docs_from_domain(conn, domain_id)
    db.close_connection(conn)

    for _doc_id, _domain_id, uri, _doc_title, doc_text in rows:
        print("********************************")
        print(uri)
        chunks = get_chunks_from_text(SAMPLE_DOC, chunk_maker)
        write_to_file('****************************************************************\n')
        for chunk in chunks:
            write_to_file(chunk + '\n==============\n')
        write_to_file('\n\n')

def test_chunker_single_doc():
    chunks = get_chunks_from_text(SAMPLE_DOC, chunk_maker)
    for chunk in chunks:
        write_to_file(chunk + '\n==============\n')
    
SAMPLE_DOC = """
Data Processing Steps

TO DO:
- add logging to capture links missing content, etc.
- step 2a: look for and remove irrelevant chunks
- check for empty or small pages ie from SPAs

- PREPARE -

1. Review site using inspect and establish which tag, tag id or class to spider

2. Create domain record with spider_notes with that info

3. Update the get_page_contents() to retrieve proper target

4. Update domain name and domain_id in 3 processing scripts


- RUN -

1. Step 1: spider side and populate document table
 Verify get_page_contents retrieval logic
 Set single to true
 Set domain to "https://domain.com", with no / at end
 Run script and verify:
  console shows content found in correct section (i.e. id=main)
  content written to page.txt is correct
  spider_log has no errors
 Change single to False and run Step 1 fully
 Check logfile
 Check db document table sorted by doc_uri
   find duplicate doc_text
  SELECT *
  FROM document
  WHERE domain_id = 31
  ORDER BY doc_uri   

2. Step 2: populate document_chunk from document records
set chunk_maker
set g_domain_id
run script
check logfile
check db document_chunks table
  search for long chunks
    SELECT length(chunk_text), dc.doc_chunk_id, dc.chunk_text, d.doc_uri
    FROM document_chunk dc
    JOIN document d ON dc.doc_id = d.doc_id
    WHERE domain_id = 25
    ORDER BY LENGTH(chunk_text) desc
    LIMIT 100
  search for redundant and useless chunks
    SELECT dc.*
    FROM document_chunk dc
    JOIN document d ON dc.doc_id = d.doc_id
    WHERE domain_id = 28
    ORDER BY chunk_text

3. Step 3: update Pinecone index with chunks
set domain_id
run script
check logfile
verify:
 select count(*) from document_chunk where domain_id = 22
 select count(*) from document_chunk
 compare with index count in Pinecone console

4. Create user

- TEST -

1. Login as user and test
- domain defaults to correct value
- "what does this company do" ; check chunks and response


Other tests:
check for chunks that are very long
SELECT domain_id, LENGTH(chunk_text), doc_chunk_id
FROM document_chunk dc
JOIN document d ON dc.doc_id = d.doc_id
WHERE LENGTH(chunk_text) > 2000
ORDER BY domain_id

"""

#####################################################
#clean_chunk = re.sub('\s+', ' ', chunk)