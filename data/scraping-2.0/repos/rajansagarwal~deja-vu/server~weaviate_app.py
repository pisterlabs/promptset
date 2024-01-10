import os
from flask import Flask, request, jsonify
import cohere
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()
import weaviate
import json

client = weaviate.Client(
    url = os.environ['DB_URL'],  
    auth_client_secret=weaviate.AuthApiKey(api_key=os.environ['WEAVIATE_API_KEY']), 
    additional_headers = {
        "X-Cohere-Api-Key": os.environ['COHERE_API_KEY']
    }
)

# env reference to COHERE_API_KEY
COHERE_API_KEY = os.environ['COHERE_API_KEY']

app = Flask(__name__)
CORS(app, origins='http://localhost:3000')
co = cohere.Client(os.environ['COHERE_API_KEY'])

print("finished loading")

@app.route('/api/test', methods=["POST", "GET"])
def test():
    return jsonify({'results': 'successfully loaded test endpoint'})


@app.route('/api/add', methods=["POST", "GET"])
def add():
    client.batch.configure(batch_size=20)
    with client.batch as batch:
        properties = {
            "file_name": request.form['File Name'],
            "transcription": request.form['Transcription'],
            "context": request.form['Context'],
            "people": request.form['People']
        }
        batch.add_data_object(
            data_object=properties,
            class_name="Media"
        )
    return jsonify({'results': 'successfully added new video embedding'})

@app.route('/api/branch', methods=["POST", "GET"])
def branch():
    query = request.form['query']
    print(f"Query: {query}")
    response = (
        client.query
        .get("Media", ["file_name", "transcription", "context", "people"])
        .with_near_text({"concepts": [query]})
        .with_limit(2)
        .do()
    )
    ARTICLE = []

    responses = response["data"]["Get"]["Media"]
    ARTICLE.append(f"you are a conversational AI that can search through memories, with access to transcripts and visual contexts. answer the query: '{query}', using the visual context, transcription, and people from the following clips: ")
    for r in responses:
        ARTICLE.append(f"Context:")
        ARTICLE.append(r.get("context"))
        ARTICLE.append(f"Transcription: ")
        ARTICLE.append(r.get("transcription"))
        ARTICLE.append(f"People: ")
        ARTICLE.append(r.get("people"))    
    co_summary = co.summarize(text=' '.join(ARTICLE))
    print(responses)
    return jsonify({'results': responses, 'summary': co_summary })


if __name__ == '__main__':
    app.run(debug=True, port=2000)