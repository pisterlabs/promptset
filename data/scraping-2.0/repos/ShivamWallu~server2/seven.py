import os
import urllib.parse
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext, load_index_from_storage, StorageContext
from langchain.llms import OpenAI
import logging
from dotenv import load_dotenv
import json  # Import the 'json' module

load_dotenv()

app = Flask(__name__)
DB_USERNAME = 'stagedb'
DB_PASSWORD = 'Stage@#$2023@'
encoded_password = urllib.parse.quote_plus(DB_PASSWORD)
DB_HOST = '44.217.100.212'
DB_PORT = '3306'
DB_NAME = 'stagedb'

app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+mysqlconnector://{DB_USERNAME}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    tablename = 'users'
    id = db.Column(db.Integer, primary_key=True)
    fname = db.Column(db.String(50))

    def __repr__(self):
        return f"<User(id={self.id}, fname={self.fname})>"

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 2000
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(api_key=os.environ["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"], temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    index.storage_context.persist('index.json')

def get_index():
    try:
        if not os.path.exists('index.json'):
            construct_index("Context")
    except Exception as e:
        logging.error(f"Error constructing index: {str(e)}")

@app.route('/ask_ai', methods=['GET', 'POST'])
def ask_ai():
    try:
        if request.is_json:
            query = request.json.get('query')
            if query:
                # Load the index from storage
                storage_context = StorageContext.from_defaults(persist_dir='.')
                index = load_index_from_storage(storage_context)
                response = index.query(query)
                return jsonify({'response': response.response})
            else:
                return jsonify({'error': 'Query parameter not found'})
        else:
            return jsonify({'error': 'Unsupported Media Type', 'message': 'Request Content-Type must be application/json'})
    except Exception as e:
        logging.error(f"Error processing ask_ai request: {str(e)}")
        return jsonify({'error': 'Internal Server Error'})
    
@app.route('/test_ask_ai', methods=['GET'])
def test_ask_ai():
    try:
        query = request.args.get('query', 'i am feeling sad please help me')
        response = ask_ai_function(query)

        if isinstance(response, dict):
            with app.app_context():
                db.create_all()
                user = User.query.first()
                fname = user.fname if user else None
                greeting = f"Hi {fname}," if fname else ""        
                
                return jsonify({
                    'query': query,
                    'response': {'response': greeting + response['response']},
                    'fname': fname,
                })
        else:
            return jsonify({'error': 'Invalid response'})
    
    except json.decoder.JSONDecodeError:
        return jsonify({'error': 'Failed to decode JSON response'})
    except Exception as e:
        logging.error(f"Error processing test_ask_ai request: {str(e)}")
        return jsonify({'error': 'Internal Server Error'})

def ask_ai_function(query):
    try:
        if query:
            storage_context = StorageContext.from_defaults(persist_dir='index.json')
            index = load_index_from_storage(storage_context)
            response = index.query(query)
            return {'response': response.response}
        else:
            return {'error': 'Query parameter not found'}
    except Exception as e:
        logging.error(f"Error processing ask_ai request: {str(e)}")
        return {'error': 'Internal Server Error'}

@app.route('/get_user_fname', methods=['GET'])
def get_user_fname():
    try:
        with app.app_context():
            db.create_all()
            user = User.query.first()
            fname = user.fname if user else None
            
        return jsonify({'fname': fname})
    except Exception as e:
        logging.error(f"Error retrieving user's first name: {str(e)}")
        return jsonify({'error': 'Internal Server Error'})

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    get_index()
    app.run(debug=True, host='0.0.0.0', port=8501)
