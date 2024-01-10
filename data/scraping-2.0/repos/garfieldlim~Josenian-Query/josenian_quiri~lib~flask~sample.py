from flask import Flask, jsonify, request
from datetime import datetime
import openai
import json
import numpy as np
from pymilvus import connections, Collection

app = Flask(__name__)

# Constants
OPENAI_API_KEY = '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'
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"
max_tokens = 8000
dimensions = 1536
openai.api_key = OPENAI_API_KEY

# Definitions
partition_name = 'facebook_posts'
bundled_schema = {'rmrj_articles': ['author', 'title', 'published_date', 'text'],
                  'facebook_posts': ['text', 'time', 'link'],
                  'usjr_about': ['text', 'content_id'],
                  'all': ['author', 'title', 'published_date', 'text', 'time', 'post', 'link', 'content_id']}
collection_names = bundled_schema[partition_name]
search_params = {
    "metric_type": "L2",
    "offset": 0,
}

# Connect to Milvus
# Check if the connection already exists
if connections.has_connection('default'):
    connections.remove_connection('default')  # Disconnect if it exists

# Now, reconnect with your new configuration
connections.connect(alias='default', host='localhost', port='19530')

@app.route('/search', methods=['POST'])
def search():
    # Get the question from the request body
    question = request.json['question']

    # Vectorize the question
    query_vectors = get_embedding(question)
    query_vectors = np.array(query_vectors)
    if len(query_vectors.shape) == 1:
        query_vectors = query_vectors.reshape(1, -1)

    # Perform the search
    results = perform_search(query_vectors)

    # Process the results
    final_results = process_results(results)

    # Generate the response
    response = generate_response(question, final_results)

    # Return the response as JSON
    return jsonify({'response': response})

# Function definitions
def get_embedding(text, model=embedding_model):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

def perform_search(query_vectors):
    results = []
    for name in collection_names:
        collection = Collection(f"{name}_collection")
        collection.load()
        result = collection.search(
            data=query_vectors,
            anns_field="embeds",
            param=search_params,
            limit=5,
            partition_names=[partition_name],
            output_fields=[name, 'uuid'],
            consistency_level="Strong"
        )
        results.append(result)
    return results

def process_results(results):
    # Initialize a dictionary to hold unique results
    unique_results = {}

    for i, name in enumerate(collection_names):
        for result in results[i]:
            for item in result:
                uuid = item.entity.get('uuid')
                data = {
                    'uuid': uuid,
                    name: item.entity.get(name),
                    'distance': item.distance
                }

                # If this UUID is not in the dictionary, or it is but the new distance is smaller, update the entry
                if uuid not in unique_results or item.distance < unique_results[uuid]['distance']:
                    unique_results[uuid] = data

    # Convert the dictionary back into a list of dictionaries
    results_object = list(unique_results.values())

    # Sort the results by distance
    sorted_results = sorted(results_object, key=lambda x: x['distance'])

    # Return the final results
    return sorted_results[:5]

def generate_response(prompt, database_json):
    # Format the input as per the desired conversation format
    string_json = json.dumps(database_json)
    conversation = [
        {'role': 'system', 'content': "You are Josenian Quiri. University of San Jose-Recoletos' general knowledge base assistant. Refer to yourself as JQ."},
        {'role': 'user', 'content': prompt},
        {'role': 'system', 'content': f'Here is the database JSON from your knowledge base:\n{string_json}'},
        {'role': 'user', 'content': ''}
    ]

    # Convert the conversation to a string
    conversation_str = ''.join([f'{item["role"]}: {item["content"]}\n' for item in conversation])

    # Generate the response using OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        temperature=1,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Extract the generated response from the API's response
    generated_text = response['choices'][0]['message']['content']

    # Return the response
    return generated_text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7999, debug=True)