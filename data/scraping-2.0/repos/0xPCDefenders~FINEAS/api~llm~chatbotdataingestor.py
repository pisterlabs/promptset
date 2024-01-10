import json
import tiktoken 
import pinecone
from tqdm.auto import tqdm
from uuid import uuid4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask import Flask, request, jsonify
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
import os
from dotenv import load_dotenv
import urllib.parse
import hashlib
from datasets import Dataset
import datetime

app = Flask(__name__)

load_dotenv()
PASS_KEY = os.environ.get("PASS_KEY")
OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")
openai.api_key = OPEN_AI_API_KEY
MODEL = 'text-embedding-ada-002'
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

@app.route('/ingestor', methods=['POST'])
def ingest_data():
    #get params
    raw_data = request.form.get("info")
    passhash = (request.headers.get('Authorization'))[7:]
    #security measures
    sha256_hash = hashlib.sha256()
    sha256_hash.update(PASS_KEY.encode('utf-8'))
    HASH_KEY = sha256_hash.hexdigest()
    if passhash != HASH_KEY:
        return jsonify({'error': 'Unauthorized access'}), 401
    
    if not raw_data:
        print("Raw Data: ", raw_data)
        return jsonify({'error': 'No data given'}), 400
    
    raw_data = urllib.parse.unquote(raw_data)  # Decode URL-encoded prompt
    #prepare & ingest data
    list_dict = []
    data = json.loads(raw_data)

    # Iterate over the JSON data and create dictionaries
    i = 0
    # Get the current date
    current_date = datetime.date.today()
    # Format the date as a string in "YYYY-MM-DD" format
    formatted_date = current_date.strftime("%Y-%m-%d")
    for info_key, info in data.items():
        dict_item = {'id':i, 'info': info_key, 'current_date': formatted_date, 'text': info}
        list_dict.append(dict_item)
    dataset = Dataset.from_list(list_dict)
    print(dataset[0])

    embed = OpenAIEmbeddings(
    document_model_name=MODEL,
    query_model_name=MODEL,
    openai_api_key=OPEN_AI_API_KEY
    )
    
    index_name = 'langchain-retrieval-augmentation'

    pinecone.init(
            api_key=PINECONE_API_KEY,  # find api key in console at app.pinecone.io
            environment="gcp-starter"  # find next to api key in console
    )

    # we create a new index

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
                name=index_name,
                metric='dotproduct',
                dimension=1536 # 1536 dim of text-embedding-ada-002
        )
    index = pinecone.Index(index_name)

    batch_limit = 1

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1,
    chunk_overlap=1,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
    )

    print(dataset)
    
    for i, record in enumerate(tqdm(dataset)):
        texts = []
        metadatas = []
        # Get metadata fields for this record
        metadata = {
            'id': str(record['id']),
            'info': str(record['info']),
            'current_date': str(record['current_date']),
            'text': str(record['text'])  # Ensure 'text' is a string
        }
        # Now create chunks from the record text
        text_splitter._chunk_size = len(str(record['text']))
        record_texts = text_splitter.split_text(str(record['text']))
        # Create individual metadata dicts for each chunk
        record_metadatas = [{
            "chunk": j,
            **metadata
        } for j, text in enumerate(record_texts)]
        # Append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        # If we have reached the batch_limit, we can add texts
        if len(texts) >= batch_limit:
            print(texts)
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas))
            texts = []
            metadatas = []
        print(index.describe_index_stats())

    return "200 Status OK"

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding('p50k_base')
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=6001, debug=True)