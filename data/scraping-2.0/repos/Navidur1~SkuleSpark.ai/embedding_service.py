import pinecone
import openai
from database import insert_one, get_data_one, update_one, pc_get_many, pc_insert_one
from flask import Blueprint, request, json, jsonify
from bson.objectid import ObjectId
import os
from dotenv import load_dotenv, find_dotenv

embedding_service = Blueprint('embedding_service', __name__)

# Set up open ai 
dotenv_path = find_dotenv(raise_error_if_not_found=True)
load_dotenv(dotenv_path)
client = openai.OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

embedding_model = "text-embedding-ada-002"

@embedding_service.route('/confirm_results', methods=["POST"])
def create_embeddings():
    data = request.json

    # Check if 'file_id' and 'elements' exists in the data dictionary
    if 'file_id' not in data:
        return "No 'file_id' found in the provided data.", 400
    
    if 'confirmed_elements' not in data:
        return "No 'confirmed_elements' found in the provided data.", 400
    
    file_id = data['file_id']
    confirmed_elements = data['confirmed_elements']

    success = chunk_elements(confirmed_elements, file_id)

    if not success:
        return "Could not chunk elements.", 400

    success, data = get_data_one('Files', {'_id': ObjectId(file_id)}, {'chunk_ids': 1})

    if not success:
        return "Could not retrieve file", 400
    
    chunks = data['chunk_ids']

    for chunk_id in chunks:
        text_to_embed = ""
        success, data = get_data_one('Chunks', {'_id': chunk_id}, {'element_ids': 1, 'text': 1})

        text_to_embed = data['text']
        embedding_res = client.embeddings.create(input=text_to_embed, model=embedding_model)
        pinecone_entry = (str(chunk_id), embedding_res.data[0].embedding, {"file_id": file_id})
        success = pc_insert_one([pinecone_entry])
        if not success:
            return "Error"
        
    return jsonify("Embed success"), 200

def chunk_elements(elements, file_id):
    chunks = []
    threshold = 512
    stored_ids = []
    cur_element_ids = []
    cur_text = ""
    chunk_success = True

    for i in range(len(elements)):
        # Update element in mongoDB
        data = elements[i]
        chunk_success, error_message = update_one('Elements', {'_id': ObjectId(data['id'])}, {'$set': {'text': data['text']}})
                
        if not chunk_success:
            print(error_message)
            break

        stored_ids.append(data['id'])

        # Over the threshold
        if len(cur_text) + len(data['text']) >= threshold:
            chunks.append({'element_ids': cur_element_ids, 'text': cur_text})
            cur_element_ids = []
            cur_text = ""
            
        cur_text += data['text']
        cur_element_ids.append(data['id'])
    
    if chunk_success:
        if len(cur_element_ids) > 0:
            chunks.append({'element_ids': cur_element_ids, 'text': cur_text})

        chunk_ids = []
        for chunk in chunks:
            chunk_success, chunk_id = insert_one('Chunks', chunk)
            chunk_ids.append(chunk_id)

        chunk_success, error_message = update_one('Files', {'_id': ObjectId(file_id)}, {'$set': {'chunk_ids': chunk_ids}})
        
        if not chunk_success:
            print(error_message)
            
    return chunk_success


