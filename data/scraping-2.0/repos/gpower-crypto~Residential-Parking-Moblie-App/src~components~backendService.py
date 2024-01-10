from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5000"]}})  # CORS configuration

# Replace these values with your actual API keys
HF_API_TOKEN = "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"
CLAUDE_API_KEY = "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"

CHROMADB_PATH = './database/chroma'
METADATA_DB_PATH = './database/metadata.json'
TEMP_PDF_FILE_NAME = 'temp.pdf'
CHUNK_SIZE = 1000

embedder = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
)
chroma_client = Chroma(
    embedding_function=embedder,
    persist_directory=CHROMADB_PATH
)
claude = Anthropic(api_key=CLAUDE_API_KEY)

def claude_completion(prompt, model_name="claude-instant-1", max_tokens=500):
    location_prompt = f"{HUMAN_PROMPT} " + str(prompt) + f" {AI_PROMPT}"
    res = claude.completions.create(
        model=model_name,
        max_tokens_to_sample=max_tokens,
        prompt=location_prompt
    )
    return res.completion


def parse_location_from_response(response):
    try:
        # Try to parse the response as JSON
        response_json = json.loads(response)

        # Assuming the response is a JSON object, extract the location
        location = response_json.get('location', None)
        return location
    except json.JSONDecodeError:
        # If parsing as JSON fails, treat the response as a plain string
        # Modify this part based on the actual structure of your response
        location = response.strip()  # Assuming location is a string in this case
        return location
    except Exception as e:
        print(f"Unexpected error in parse_location_from_response: {str(e)}")
        return None

def get_location_from_query(query):
    try:
        # Provide a prompt with a placeholder for the location in the answer
        prompt = """
        Context: User is asking about parking locations. First verify if the location is valid and if it is within singapore. If both are ture then, return only the latitude and longitude values alone. 
        Remember to answer only with the values of the latitude and longitude or leave it empty.

        Question: Where is the {location} parking area?
        Answer: {latitude}, {longitude}
        """.format(location=query, latitude=None, longitude=None)

        # Use Anthropic to complete the prompt
        response = claude_completion(prompt)

        # Parse the response to extract the location
        location = parse_location_from_response(response)

        return location
    except Exception as e:
        print(f"Error in get_location_from_query: {str(e)}")
        return None

@app.route('/extract_location', methods=['POST'])
def extract_location():
    data = request.get_json()
    print("Received data:", data)

    query = data.get('query')
    # Additional logic using Chroma, Hugging Face, Anthropic, etc.
    # For now, let's just get the location from the Cloud AI API
    location = get_location_from_query(query)
    
    return jsonify({'location': location})

if __name__ == '__main__':
    app.run(debug=True, host='localhost')
