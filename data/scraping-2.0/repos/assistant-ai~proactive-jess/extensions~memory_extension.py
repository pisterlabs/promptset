from openai import OpenAI
import pinecone
import os
import json
import uuid
from dotenv import load_dotenv
from .jess_extension import jess_extension
from rest.token_management import get_user_id
from rest.main_app import app
from quart import jsonify, request


load_dotenv()


PINECONE_KEY = os.getenv('PINECONE_KEY')
USER_ID = os.getenv('USER_ID')

client = OpenAI()


pinecone.init(api_key=PINECONE_KEY, environment='gcp-starter')


class Memory(object):

    def __init__(self, user_id, index):
        self.user_id = user_id
        self.index = index
    
    def _embed_text_with_openai(self, text):
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    @jess_extension(
        description="Store a new memory/fact about a user so you can retrive it later",
        param_descriptions={
            "fact": "A memory or a fact to store about the user"
        }
    )
    def store_in_long_term_memory(self, fact: str):
        # Generate a unique identifier for the fact (e.g., UUID)
        fact_id = str(uuid.uuid4())

        # Generate the fact vector
        fact_vector = self._embed_text_with_openai(fact)

        # Store the fact with user ID in the metadata
        self.index.upsert(vectors=[{
            "id": fact_id, 
            "values": fact_vector, 
            "metadata": {
                "user_id": self.user_id,
                "fact": fact
            }
        }])
        return "DONE"

    @jess_extension(
        description="Retrive relevant memory/facts about user by askin a questiont",
        param_descriptions={
            "question": "Query/question to use to retrieve relevant memory/facts about user",
            "count": "Amount of facts/memories to return (sorted by relevance), default is 5"
        }
    )
    def query_from_long_term_memory(self, question: str, count: int, json_dump=True):
        if count <= 0:
            count = 5
        # Assuming 'query_vector' is the vector representation of your query
        query_vector = self._embed_text_with_openai(question)

        # Define a filter to only include keys that start with the user's ID
        query_filter = {
            "user_id": self.user_id
        }

        # Perform the query
        result = self._extract_sorted_facts(self.index.query(filter=query_filter, top_k=count, vector=query_vector, include_metadata=True))
        if json_dump:
            return json.dumps(result)
        else:
            return result

    def _extract_sorted_facts(self, results_dict):
        # Extracting the facts and their scores
        facts_with_scores = [(match['metadata']['fact'], match['score']) for match in results_dict['matches']]

        # Sorting the facts by score in descending order
        sorted_facts = sorted(facts_with_scores, key=lambda x: x[1], reverse=True)

        # Extracting only the facts from the sorted tuples
        sorted_fact_strings = [fact for fact, score in sorted_facts]
        return sorted_fact_strings

    @staticmethod
    def create_memory_extension(user_id=USER_ID):
        index_name = 'user-facts-index'
        index = None
        try:
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(index_name, dimension=1536)  # Adjust dimension based on the model
            index = pinecone.Index(index_name)
        except:
            pass
        return Memory(user_id, index)


@app.route('/memories/', methods=['POST'])
async def add_memory():
    data = await request.get_json(force=True)

    if 'memory' not in data:
        return jsonify({'error': 'Bad Request', 'message': 'Your request is missing "memeory"'}), 400

    memory = data['memory']
    token = request.headers.get('Authorization').split(' ')[1]
    user_id = None
    try:
        user_id = get_user_id(token)
    except:
        return jsonify({"error": "please re-authentificate"}), 500
    if not user_id:
        return jsonify({"error": "please re-authentificate"}), 500
    
    memory_client = Memory.create_memory_extension(user_id=user_id)
    memory_client.store_in_long_term_memory(memory)

    return jsonify({'message': 'Memory added successfully'}), 201


@app.route('/memories/', methods=['GET'])
async def get_memories():
    # Extract the token from the Authorization header
    user_id = None
    if 'Authorization' not in request.headers:
        return jsonify({"error": "please re-authentificate"}), 500
    token = request.headers.get('Authorization').split(' ')[1]
    try:
        user_id = get_user_id(token)
    except:
        return jsonify({"error": "please re-authentificate"}), 500
    if not user_id:
        return jsonify({"error": "please re-authentificate"}), 500
    memory_client = Memory.create_memory_extension(user_id=user_id)
    query = request.args.get("question")
    count = 0
    if "count" in request.args:
        count = int(request.args.get("count"))

    response = {
        "memories": memory_client.query_from_long_term_memory(query, count, json_dump=False)
    }

    return jsonify(response), 200
