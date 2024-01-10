# Import necessary libraries
from flask import Flask, request, jsonify, render_template
import os
import logging
import pinecone
import base64
import openai
import google.generativeai as genai

# Initialize Flask app
app = Flask(__name__)

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Environment configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_KEY = os.getenv('GEMINI_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Instantiate OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Configure other APIs
genai.configure(api_key=GEMINI_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment='gcp-starter')

# Generative model configuration
generation_config = {
    "temperature": 0.6,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]

# Initialize the generative model
model = genai.GenerativeModel(
    model_name="gemini-pro-vision",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Define utility functions
def input_image_setup(file):
    try:
        # Ensure the file is not empty
        if file is None or file.tell() == 0:
            logging.error("No file provided or file is empty.")
            return None

        # Reset file pointer to the start
        file.seek(0)

        # Encode the file data as base64
        encoded_data = base64.b64encode(file.read()).decode('utf-8')

        return {
            "mime_type": "image/jpeg",  # Consider dynamic MIME type detection for different image formats
            "data": encoded_data
        }

    except Exception as e:
        logging.error(f"Error encoding image: {e}")
        return None

def get_embedding(response_text):
    """
    Retrieves the embedding for a given text using OpenAI's embedding model.

    Args:
        response_text (str): The text for which the embedding is to be generated.

    Returns:
        list: A list of floats representing the embedding of the text.
        Returns None if an error occurs during the embedding process.

    """
    if not response_text:
        logging.error("No response text provided for embedding.")
        return None

    try:
        embedding_response = client.embeddings.create(model="text-embedding-ada-002", input=response_text)

        # Check if the response contains the expected data
        if 'data' in embedding_response and len(embedding_response['data']) > 0:
            embedding_data = embedding_response['data'][0]['embedding']
            return embedding_data
        else:
            logging.error("Invalid response structure from embedding API.")
            return None

    except Exception as e:
        logging.error(f"Error in generating embedding: {e}")
        return None



def handle_request_error(e, action):
    logging.error(f"Error {action}: {e}")
    return jsonify({"error": str(e)}), 500

def initialize_pinecone_index(index_name):
    if index_name in pinecone.list_indexes():
        return pinecone.Index(index_name)
    else:
        raise ValueError(f"Index '{index_name}' does not exist.")


# Define routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        image = request.files.get("image")
        if not image:
            return jsonify({"error": "No image provided"}), 400

        input_prompt = "You are an expert in identifying images and objects in the image and describing them."
        question = "Describe this picture and identify it in less than 100 words:"
        image_prompt = input_image_setup(image)
        prompt_parts = [input_prompt, image_prompt, question]

        response = model.generate_content(prompt_parts)
        embedding = get_embedding(response.text)

        return jsonify({"embedding": embedding})
    except Exception as e:
        return handle_request_error(e, "generation")

@app.route("/search", methods=["POST"])
def search():
    try:
        query = request.json.get("query")
        index = initialize_pinecone_index("imgcompare")
        query_results = index.query(vectors=[query], top_k=10)
        return jsonify({"results": query_results})
    except Exception as e:
        return handle_request_error(e, "searching embeddings")

@app.route("/upsert", methods=["POST"])
def upsert():
    try:
        image, image_id = request.files.get("image"), request.form.get("image_id")
        if not image or not image_id:
            return jsonify({"error": "Image and image ID are required"}), 400

        image_prompt = input_image_setup(image)
        prompt_parts = [image_prompt]

        try:
            response = model.generate_content(prompt_parts)
        except Exception as e:
            return handle_request_error(e, "generate_content")

        embedding = get_embedding(response.text)

        # New code: Check each element in embedding to confirm it's a float
        if all(isinstance(item, float) for item in embedding):
            logging.info("All elements in embedding are floats.")
        else:
            non_floats = [type(item) for item in embedding if not isinstance(item, float)]
            logging.error(f"Non-float types in embedding: {set(non_floats)}")
            
        # Add logging for image_id
        logging.info(f"Image ID type: {type(image_id)}, Image ID: {image_id}")
        
        # Test upsert with hardcoded values
        test_embedding = [0.1, 0.2, 0.3]  # Example simple embedding
        test_image_id = "test_id"  # Example simple image ID
        try:
            index = initialize_pinecone_index("img-compare")
            index.upsert(vectors={test_image_id: test_embedding})
            logging.info("Test upsert successful")
        except Exception as e:
            logging.error(f"Test upsert error: {e}")
      
        try:
            index = initialize_pinecone_index("img-compare")
            index.upsert(vectors={image_id: embedding})
        except Exception as e:
            return handle_request_error(e, "upsert")

        return jsonify({"message": "Image upserted successfully"})
    except Exception as e:
        return handle_request_error(e, "upsert")



# Run the app
if __name__ == "__main__":
    app.run(debug=False)  # Set debug to False for production
