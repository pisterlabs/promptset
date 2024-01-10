from flask import Flask, request, jsonify
import openai
import firebase_admin
from firebase_admin import storage, credentials
from dotenv import dotenv_values
from flask_cors import CORS
import requests
import openai 
from dotenv import dotenv_values

app = Flask(__name__)
CORS(app)

env_vars = dotenv_values('./env')

openai.api_key = env_vars.get('key')
bucket = env_vars.get('bucket')
doc_name = env_vars.get('doc_name')
cert = env_vars.get('cert')

# Initialize Firebase
cred = credentials.Certificate(cert)  # Add your Firebase credentials
firebase_admin.initialize_app(cred, {
    'storageBucket': bucket
})
bucket = storage.bucket(doc_name)

@app.route('/',methods = ['GET', 'POST'])
def proof_of_life():
    return "HELLO BUDDY CONGNIFUSE_AI IS ALIVE"
    
def download_document(document_name):
    blob = bucket.blob(document_name)
    # Download the file from Firebase
    file_contents = blob.download_as_string()
    return file_contents.decode("utf-8") if file_contents else None

@app.route('/answerquestions', methods=['POST'])
def answer_document_questions():
    user_input = request.form['user_input']
    document_name = request.form['document_name']  # Assuming this is the name of the file in Firebase Storage
    # return jsonify({"chatbot_response": "Your response here"}), 200

    # Download document content from Firebase Storage
    document_content = download_document(document_name)


    if document_content:
        prompt = f"Document: {document_content}\nUser: {user_input}\nChatbot:"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
        )
        return jsonify({"chatbot_response": response['choices'][0]['text'].strip()}), 200
    else:
        return jsonify({"error": "No document available. Please upload a document first."}), 404
    
    
    # topic.py

@app.route('/combined_learning/<topic>', methods=['GET'])  # Combined endpoint
def combined_learning(topic):
    try:
        # Get the learning program from OpenAI
        learning_program = create_learning_program(topic)

        # Get content from Wikipedia
        content_response = requests.get(
            f"https://en.wikipedia.org/w/api.php?action=query&format=json&titles={topic}&prop=extracts&exintro=1"
        )
        content_response.raise_for_status()  # Raise an exception for HTTP errors
        content = content_response.json().get('query', {}).get('pages', {}).get(next(iter(content_response.json().get('query', {}).get('pages', {}))), {}).get('extract')

        # Return a combined response
        return jsonify({
            "learning_program": learning_program,
            "wikipedia_content": content
        }), 200

    except Exception as e:
        return jsonify({"error": f"Error retrieving combined response: {e}"}), 500


@app.route('/learning/<topic>', methods=['GET']) #endpoint is working
def create_learning_program(topic):
    prompt = f"Create a personalized learning program on {topic}. Include sections on introduction, key concepts, examples, practice exercises, and conclusion."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300,
        temperature=0.7,
    )
    learning_program = response['choices'][0]['text'].strip()
    # learning_program = "This is a sample learning program."

    return learning_program


@app.route('/learningprogram', methods=['GET'])
def get_learning_program(topic=None):
    if not topic:
        topic = request.args.get('topic')
    
    if not topic:
        return jsonify({"error": "Topic not provided"}), 400
    
    learning_program = create_learning_program(topic)
    
    return jsonify({"learning_program": learning_program}), 200


# @app.route('/learningprogram', methods=['GET']) #endpoint is not working
# def get_learning_program():
#     # Extract the topic from the query parameters or form data
#     topic = request.args.get('topic')
    
#     if not topic:
#         return jsonify({"error": "Topic not provided"}), 400
    
#     # Generate the learning program for the given topic
#     learning_program = create_learning_program(topic)
    
#     # Return the learning program as a JSON response
#     return jsonify({"learning_program": learning_program}), 200

# def get_program_section(learning_program, user_selection):
#     sections = learning_program.split('\n')[1:-1]
#     try:
#         selected_section = sections[int(user_selection) - 1]
#         return selected_section
#     except (ValueError, IndexError):
#         return None



@app.route('/getcontent/<topic>', methods=["POST"])
def get_content(topic):
    try:
        wikipedia_api_url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&titles={topic}&prop=extracts&exintro=1"
        response = requests.get(wikipedia_api_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        data = response.json()

        if "query" in data and "pages" in data["query"]:
            page = next(iter(data["query"]["pages"].values()))
            if "extract" in page:
                content = page["extract"]
                return jsonify({"content": content}), 200

        return jsonify({"message": "No content found for the given topic"}), 404

    except requests.exceptions.RequestException as req_error:
        return jsonify({"error": f"Error making Wikipedia API request: {req_error}"}), 500

    except Exception as e:
        return jsonify({"error": f"Error fetching content from Wikipedia: {e}"}), 500
    
# @app.route("/alt/content/<topic>", methods="GET")
def fetch_alternative_content_1(topic):
    try:
        # Use the Wikipedia API to fetch information about the topic
        wikipedia_api_url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&titles={topic}&prop=extracts&exintro=1"
        response = requests.get(wikipedia_api_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        data = response.json()

        # Check if the API response contains an 'extract' field
        if "query" in data and "pages" in data["query"]:
            page = next(iter(data["query"]["pages"].values()))
            if "extract" in page:
                content = page["extract"]
                return content

    except requests.exceptions.RequestException as req_error:
        print(f"Error making API request for alternative content 1: {req_error}")
    except Exception as e:
        print(f"Error fetching alternative content 1: {e}")
    return None

# @app.route('/combined_endpoint/<topic>', methods=['GET'])
# def combined_endpoint(topic):
#     learning_program = create_learning_program(topic)
#     content_response = get_content(topic)
#     content = content_response.json().get('content', None)
#     alternative_content = fetch_alternative_content_1(topic)
    
#     return jsonify({
#         "learning_program": learning_program,
#         "wikipedia_content": content,
#         "alternative_content": alternative_content
#     }), 200


@app.route("/combined/learningProgram/<topic>", methods=["GET"])
def all_learning(topic):
    my_program = create_learning_program(topic)
    learn_response = get_learning_program(topic)
    top_response = get_content(topic)
    cont = fetch_alternative_content_1(topic)

    return jsonify({
        "learning_program": my_program,
        "wikipedia_content": learn_response.get('learning_program', None),
        "alternative_content": top_response.get('content', None),
        "content": cont
    }), 200

# file.py

# Initialize Firebase app
try:
    # Try to get the default app, which will throw an exception if it doesn't exist
    default_app = firebase_admin.initialize_app()
except ValueError:
    # If the default app already exists, do nothing
    pass

# If the default app doesn't exist, initialize it
if not firebase_admin._apps:
    cred = credentials.Certificate(cert)
    firebase_admin.initialize_app(cred, {
        'storageBucket': doc_name
    })

@app.route('/dropFiles', methods=['POST'])
def store_file():
    try:
        uploaded_file = request.files['file']
        if uploaded_file:
            bucket = storage.bucket('bucket')
            blob = bucket.blob(uploaded_file.filename)
            blob.upload_from_file(uploaded_file)
            return jsonify({"message": "File stored successfully!"}), 200
        else:
            return jsonify({"message": "No file provided."}), 400
    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

# endpoint for getting the file stored in the storage.bucket
@app.route('/listFiles', methods=['GET'])
def list_files():
    bucket = storage.bucket('bucket')  # Access the default storage bucket
    blobs = bucket.list_blobs()  # Retrieve a list of blobs (files) in the bucket

    file_list = [blob.name for blob in blobs]  # Extracting file names from the blobs

    return jsonify({"files": file_list}), 200

if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=5000, debug=True)
    app.run()
