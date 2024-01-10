from flask import Flask, request, jsonify
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.vectorstores import Chroma
from ingest import main
from flask_cors import CORS, cross_origin
import os
import time
import shutil
import openai
import glob
from dotenv import load_dotenv
import sentry_sdk
from flask import Flask
from sentry_sdk.integrations.flask import FlaskIntegration


load_dotenv()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000"
]

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

sentry_sdk.init(
    dsn="https://f23424f295d8e523993eec840fee97d0@o1145044.ingest.sentry.io/4505755832549376",
    integrations=[
        FlaskIntegration(),
    ],

    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0
)

app = Flask(__name__)

CORS(app, allow_headers="*", send_wildcard=True)
app.config['CORS_HEADERS'] = 'Content-Type'

openQA = None


openai.api_key = os.getenv("OPENAI_API_KEY")

def get_moderation(question):
  """
  Check if the question is safe to ask the model

  Parameters:
    question (str): The question to check

  Returns a list of errors if the question is not safe, otherwise returns None
  """

  errors = {
        "hate": "Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.",
        "hate/threatening": "Hateful content that also includes violence or serious harm towards the targeted group.",
        "self-harm": "Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.",
        "sexual": "Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).",
        "sexual/minors": "Sexual content that includes an individual who is under 18 years old.",
        "violence": "Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.",
        "violence/graphic": "Violent content that depicts death, violence, or serious physical injury in extreme graphic detail.",
  }
  response = openai.Moderation.create(input=question)
  if response.results[0].flagged:
      # get the categories that are flagged and generate a message
      result = [
        error
        for category, error in errors.items()
        if response.results[0].categories[category]
      ]
      return result
  return None

def delete_vectorstore(persist_directory):
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
            return True
        except Exception as e:
            print(f"Error deleting vector store: {e}")
            return False
    else:
        return False

def check_vectorstore(persist_directory):
    if os.path.exists(persist_directory):
        print(f"Local DB exists")
        return True
    else:
        print(f"Local DB does not exist")
        return False

def build_bot(persist_directory):
    global openQA
    if os.path.exists(persist_directory):
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
        try:
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, verbose=False)
            openQA = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= False)
            return True
        except Exception as e:
            print(f"Error building the GPT4All LLM: {e}")
            return False
    else:
        return False

# Route to trigger the ingest logic
@app.route('/run_ingest', methods=['POST'])
@cross_origin()
def run_ingest():
    try:
        main()  # Call the main function with your logic
        build_bot(persist_directory)
        return jsonify({"message": "Files ingest executed and LLM built successfully"})
    except Exception as e:
        return jsonify({"error": "There was a fail during the ingestion of files or the building of the LLM: " + str(e)}), 500

@app.route('/delete-vectorstore', methods=['POST'])
def delete_vectorstore_route():
    if persist_directory:
        success = delete_vectorstore(persist_directory)
        if success:
            return jsonify({"message": "Vector store deleted successfully."}), 200
        else:
            return jsonify({"message": "Error deleting vector store."}), 500
    else:
        return jsonify({"message": "Missing 'persist_directory' parameter."}), 400

@app.route('/localdbcheck', methods=['POST'])
def localdbcheck_route():
    if persist_directory:
        success = check_vectorstore(persist_directory)
        if success:
            return jsonify({"message": "Local DB created."}), 200
        else:
            return jsonify({"message": "No Local DB created."}), 500
    else:
        return jsonify({"message": "Missing 'persist_directory' parameter."}), 400

@app.route('/debug-sentry', methods=['POST'])
def trigger_error():
    division_by_zero = 1 / 0
    return jsonify({"message": "Error triggered successfully."}), 200

@app.route('/moderate-question', methods=['POST'])
def moderate_question():
    try:
        data = request.get_json()
        question = data
        if question is None:
            return jsonify({"error": "Question not provided"}), 400
        moderation_errors = get_moderation(question)
        if moderation_errors:
            return jsonify({"errors": moderation_errors}), 400
        else:
            return jsonify({"message": "Question is safe"}), 200
    except Exception as e:
        return jsonify({'error in moderate-question route': str(e)}), 500    

@app.route('/ask-bot', methods=['POST'])
def ask_bot():
    try:
        data = request.get_json()  # Get the JSON data from the request body
        query = data  # Assuming the query is passed as a key in the JSON data
        print(query)
        print(openQA)

        if query and openQA:
            print("\n\n> Question:")
            print(query)
            # Call the 'qa' function to get the response
            res = openQA(query)
            answer = res['result']
            print("Answer:")
            print(answer)

            # Building the response JSON
            response = {
                'data': answer
            }
            return jsonify(response), 200  # Return the response with a success status code
        else:
            return jsonify({'error': 'Question not provided to backend'}), 400  # Bad request if no query provided
    except Exception as e:
        return jsonify({'error in ask-bot route': str(e)}), 500 


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
