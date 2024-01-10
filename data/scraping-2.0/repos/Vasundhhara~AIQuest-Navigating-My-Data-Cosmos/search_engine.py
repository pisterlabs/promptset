from flask import Flask, request, jsonify
import pinecone
import openai
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:8000"])


pinecone.init(api_key="YOUR_API_KEY")
openai.api_key = "YOUR_API_KEY"
index = pinecone.Index(index_name="messages")

def get_embedding(messages: str, model="text-embedding-ada-002"):
   messages = messages.replace("\n", " ")
   return openai.Embedding.create(input = [messages], model=model)['data'][0]['embedding']


@app.route("/semantic_search", methods=["GET"])
def semantic_search():
    query = request.args.get("query")
    embedded_query = get_embedding(query)
    results = index.query(queries=[embedded_query],top_k=5)
    return jsonify(results[0])  # Return the search results as JSON

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


