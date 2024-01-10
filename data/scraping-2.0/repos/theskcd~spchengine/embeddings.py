# This file will help us create embeddings from the video transcript we are providing
# We can then use a vector databse to do semantic search on top of it

import os
import openai
import pinecone

# Initialize firebase bullshit here
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
cred = credentials.Certificate("/Users/skcd/spchengine/lex_friedman/creds.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


openai.api_key = "PUT TOKEN HERE"
pinecone.init(api_key="PUT TOKEN HERE", environment="us-west1-gcp")
PINECONE_INDEX = pinecone.Index('testing-index2') # this is the testing index we have

# Finish this function as well
# def create_embedding(text: str): 

# Read the file as input to create the embeddings
# The format for this one is:
# Speaker (timestamp)
# \n
# Content
# So we want to group it together based on the timestamp 
def read_file_and_split_it(file_name: str):
    contents = []
    with open(file_name, 'r') as f:
        contents = f.readlines()
    # now loop over the contents and try to get them in the format we want which is
    # speaker : content (and the timestamp here)
    end_index = len(contents) - 1
    start_index = 0
    refined_content = []
    while start_index < end_index:
        content_line_now = ""
        # The first line is for the speaker so get that
        content_line_now = contents[start_index]
        content_line_now = content_line_now + ":" + contents[start_index + 2] + "\n"
        start_index = start_index + 4
        refined_content.append(content_line_now)
    return refined_content

# Takes as input array of content and creates embeddings
# Lets be naive for now and do stupid things that work (maybe lol)
# we take the conversation and group it in 2 instances of conversation to keep some
# context (in the future we can make this better)
def prepare_for_embeddings(contents):
    grouped_convo = []
    start_index = 0
    end_index = len(contents) - 1
    while start_index <= end_index:
        index_now = start_index
        index_limit = min(end_index, index_now + 3)
        convo_context_now = []
        while index_now <= index_limit:
            convo_context_now.append(contents[index_now])
            index_now = index_now + 1
        start_index = index_now
        grouped_convo.append("\n".join(convo_context_now))
    return grouped_convo

# Fingers crossed this works
def put_embedding_in_firestore(embedding, firestore_path):
    PINECONE_INDEX.upsert([(firestore_path, embedding)])

# This gives us back the vector of embeddings which we should use for cosine-similarity
# I think it returns a vector because it might be splitting the input text by itself
# into chunks (most probably but not sure, lets better be safe as the api works based
# on tokens we are inserting)
def openai_embedding(content):
    response = openai.Embedding.create(input=content, model="text-embedding-ada-002")
    print(dir(response))
    return response['data'][0]['embedding']

def save_to_firestore(content):
    collection = db.collection('lex_friedman')
    response = collection.add({'content': content})
    # We will also use this as the vector id when adding to pinecone DB
    path_in_firestore = response[1].path
    # We first need to store the content in firestore and get the path to the
    # firestore which will be our vector-id which we use in pinecone
    return path_in_firestore

if __name__ == '__main__':
    file_contents = read_file_and_split_it('/Users/skcd/spchengine/lex_friedman/recordings/lex_john_carmack_001.txt')
    print(len(file_contents))
    # print(file_contents[0])
    # print(file_contents[1])
    # print(file_contents[3])
    prepared_convo = prepare_for_embeddings(file_contents)
    # print(prepared_convo[0])
    # generate_embedding('skcd is awesome')
    for convo in prepared_convo:
        embedding_for_sample = openai_embedding(convo)
        firestore_path = save_to_firestore(convo)
        put_embedding_in_firestore(embedding_for_sample, firestore_path)

