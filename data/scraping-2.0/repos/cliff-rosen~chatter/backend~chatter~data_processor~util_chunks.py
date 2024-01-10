from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import re
import os
import sys
sys.path.append('db')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import db
import local_secrets as secrets

MIN_CHUNK_LENGTH = 20

def get_chunks_from_text_2(text):
    print("chunker 2")
    chunks = []
    fragments = []

    # clean input
    text.strip()
    text = re.sub('\s{3,}', '\n\n', text)    

    # built array of fragments by nn
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

def write_text_to_file(file_path, text):
    with open(file_path, 'w') as new_file:
        #clean_chunk = re.sub('\s+', ' ', chunk_text)
        #clean_chunk = clean_chunk.encode(encoding='ASCII',errors='ignore').decode()
        new_file.write(text)

def write_chunks_to_file(file_path, chunks):
    with open(file_path, 'w') as new_file:
        for chunk in chunks:
            #clean_chunk = re.sub('\s+', ' ', chunk)
            chunk = chunk.encode(encoding='ASCII',errors='ignore').decode()
            new_file.write(chunk + "\n------------------\n")

def run():
    # runtime settings
    doc_id = 5758

    text = db.get_document(doc_id)[0]["doc_text"]
    text = text.strip()
    write_text_to_file("p1.txt", text)

    chunks = get_chunks_from_text_2(text)
    write_chunks_to_file("p1-c1.txt", chunks)

text = ""    
chunks = get_chunks_from_text_2(text)
for chunk in chunks:
    print("-------------------------------")
    print(chunk)





