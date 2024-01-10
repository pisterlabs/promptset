from flask import Flask, request
import weaviate
import os
import openai
from uuid import uuid4
from langchain.text_splitter import TokenTextSplitter

# OpenAI API key configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Weaviate keys configuration
auth_config = weaviate.AuthApiKey(api_key="")
client = weaviate.Client(auth_client_secret=auth_config, url="")

# Embedding Model
embed_model = "text-embedding-ada-002"

# Weaviate Class creation
class_obj = {
    "class": "Knowledge",
    "vectorizer": "text2vec-openai",
}
client.schema.create_class(class_obj)


# Function to create embeddings
def create_embeddings(texts):
  res = openai.Embedding.create(input=texts, engine=embed_model)
  return res['data'][0]['embedding']


app = Flask(__name__)


@app.route('/webhook', methods=['POST'])
def webhook():
  data = request.json

  # Extract the URL and code values
  url_value = data['data'][0]['URL']
  code_value = data['data'][2]['code']

  # Combine the URL and code values
  combined_data = {"URL": url_value, "code": code_value}

  # Text Chunking
  text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=10)
  input_chunks = text_splitter.split_text(combined_data['code'])

  # Creating Embedding and Upserting to Weaviate
  with client.batch as batch:
    for chunk in input_chunks:
      embeds = create_embeddings(chunk)
      data_object = {"text": chunk, "knowledge_name": "Knowledge"}
      response = batch.add_data_object(data_object,
                                       "Knowledge",
                                       uuid=str(uuid4()),
                                       vector=embeds)
      print("Response from Weaviate:", response)  # Print the response

  return 'OK', 200


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)
