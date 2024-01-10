#!/usr/bin/env python3

import os
from flask import Flask, request, Response
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from text_generation import Client
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader

app = Flask(__name__)

def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
    FieldSchema(name='id', dtype=DataType.INT64, descrition='ids', max_length=500, is_primary=True, auto_id=True),
    FieldSchema(name='embeddings', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=dim),
    FieldSchema(name='sentences', dtype=DataType.VARCHAR, max_length=1024)
    ]
    schema = CollectionSchema(fields=fields, description='ITDZ Prototyp text similarity search')
    collection = Collection(name=collection_name, schema=schema)

    # create IVF_FLAT index for collection.
    index_params = {
        'metric_type':'L2',
        'index_type':"IVF_FLAT",
        'params':{"nlist":1024}
    }
    collection.create_index(field_name="embeddings", index_params=index_params)
    return collection


def insert_vectors(collection, collection_name, data):
    return collection.insert(data)

def similarity_search(collection, vectors_to_search, limit=3):
    collection.load()
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 5}
    }
    results = collection.search(vectors_to_search, "embeddings", search_params, limit=limit, output_fields=["id"])
    return results

def ids_to_vectors(collection, ids):
    res = collection.query(
        expr = f"id in {ids}",
        offset = 0,
        limit = 10,
        output_fields = ["sentences", "embeddings"]
    )
    return res


def split_pdf(path):
    doc_batch = []
    for filename in os.listdir(path):
        loader = PyPDFLoader(path + "/" + filename)
        pages = loader.load_and_split()
        batches = []
        for page in pages:
            #print("=" * 50)
            page_len = 0
            batch = ""
            for i in page.page_content.split("\n"):
                if (page_len + len(i)) > 500:
                    batches.append(batch)
                    batch = ""
                    page_len = 0
                page_len += len(i)
                batch += i + " "

        doc_batch.append(batches)
    return doc_batch


def sentences2embeddings(sentences):
    embeddings = model2.encode(sentences)
    embeddings = embeddings.tolist()
    app.logger.info(type(embeddings))

    return embeddings

#
# API Routes
#

@app.route('/process_sentences', methods=['POST'])
def process_list():
    data = request.get_json()
    sentences = data.get('sentences', [])
    app.logger.info(sentences) 
    vectors = sentences2embeddings(sentences)
    #app.logger.info("VECTORS: ", type(vectors), vectors)
    
    results = insert_vectors(collection, collection_name, [vectors, sentences])
    app.logger.info(results)
    return str(results)

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    vectors = data.get('vectors', [])

    results = similarity_search(collection, vectors, limit=3)
    ids = results[0].ids
    #app.logger.info(ids)
    vectors = ids_to_vectors(collection, ids)
    app.logger.info(vectors)

    app.logger.info(results)
    results = str(results[0]).encode('utf-8')
    vectors = str(vectors).encode('utf-8')

    return vectors

@app.route('/s2s_search', methods=['POST'])
def s2s_search():
    data = request.get_json()
    sentences = data.get('sentences', [])
    app.logger.info(sentences)
    vectors = sentences2embeddings(sentences)
    #app.logger.info(vectors)
    results = similarity_search(collection, vectors, limit=3)
    ids = results[0].ids
    app.logger.info(ids)
    sentences = []
    for id_ in ids:
        sentence = ids_to_vectors(collection, [id_])
        sentences.append(sentence[0]["sentences"])
    
    sentences = str(sentences).encode('utf-8')

    return sentences


@app.route('/question_answering', methods=['POST'])
def question_answering():
    data = request.get_json()
    question = data.get('question', [])
    vectors = sentences2embeddings(question)
    #app.logger.info(vectors)
    results = similarity_search(collection, vectors, limit=3)
    ids = results[0].ids
    app.logger.info(ids)
    context = ""
    for id_ in ids:
        sentence = ids_to_vectors(collection, [id_])
        context += sentence[0]["sentences"]

    prompt = f"""<|prompter|>Benutze die folgenden Informationen als Kontext:
    {context}
    Um diese Frage auf deutsch zu beanworten: {question}
    Antwort auf deutsch:<|endoftext|><|assistant|>"""
    #prompt2 = "nenne mit drei Farben, antworte auf deutsch"
    
    #answer = client.generate(prompt, max_new_tokens=1024).generated_text
    #return answer
    def answering():
        for response in client.generate_stream(prompt, max_new_tokens=1024):
            if not response.token.special:
                yield response.token.text

    return Response(answering(), mimetype='text/plain')


@app.route('/collection_details', methods=['GET'])
def collection_details():
    c = Collection(collection_name)
    data = str(c.is_empty)
    data = {
        "schema": str(c.schema),
        "descrption": str(c.description),
        "name": str(c.name),
        "is_empty": str(c.is_empty),
        "num_entities": str(c.num_entities),
        "index": str(c.indexes)
    }

    return data

@app.route('/split_pdfs', methods=["POST"])
def split_pdfs():
    data = request.get_json()
    dir_path = data.get("dir_path", [])

    batches = split_pdf(dir_path)
    sentences = []
    for i in batches:
        sentences += i
    app.logger.info(sentences)

    vectors = sentences2embeddings(sentences)
    results = insert_vectors(collection, collection_name, [vectors, sentences])

    return str(results)


if __name__ == '__main__':
    connections.connect("default", host="localhost", port="19530")
    client = Client("http://127.0.0.1:8080")

    model2 = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    #embeddings = model.encode(sentences)
    #print(embeddings)

    collection_name = "test1"
    collection = create_milvus_collection(collection_name, 1024)

    CORS(app)

    app.run(debug=True, port=5000)

