from flask import Flask, request, jsonify
from flask_cors import CORS  # Added for CORS support
import json
import requests
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
import os
import numpy as np

app = Flask(__name__)
CORS(app)  # Initialize CORS on the Flask app
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

def load_data_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def get_agenda_items(recipe_name, data):
    agenda_items = []
    for entry in data:
        if entry['Journey Name (N)'] == recipe_name:
            agenda_items.append(entry['Agenda Items (Description)'])
    return agenda_items if agenda_items else None

def get_methods(recipe_name, data):
    methods = []
    for entry in data:
        if entry['Journey Name (N)'] == recipe_name:
            methods.append(entry['Methods (N)'])
    return methods if methods else None

def get_method_details(methods_list, data):
    method_details = []
    for method in methods_list:
        method = method.strip()  # Remove any leading/trailing whitespace
        for entry in data:
            if entry.get('Uniques') == method:
                method_details.append({
                    'method': method,
                    'description_short': entry.get('Description (short)', ''),
                    'ai_response': entry.get('AI Response', '')
                })
                break  # Stop searching after the first match
    return method_details

@app.route('/get_recipe', methods=['POST'])
def get_recipe():
    content = request.json
    recipe_name = content.get('recipe_name')
    if not recipe_name:
        return jsonify({"error": "No recipe name provided"}), 400

    tasks_url = 'https://gs.jasonaa.me/?url=https://docs.google.com/spreadsheets/d/e/2PACX-1vSmp889ksBKKVVwpaxhlIzpDzXNOWjnszEXBP7SC5AyoebSIBFuX5qrcwwv6ud4RCYw2t_BZRhGLT0u/pubhtml?gid=1980586524&single=true'
    flow_url = 'https://gs.jasonaa.me/?url=https://docs.google.com/spreadsheets/d/e/2PACX-1vSmp889ksBKKVVwpaxhlIzpDzXNOWjnszEXBP7SC5AyoebSIBFuX5qrcwwv6ud4RCYw2t_BZRhGLT0u/pubhtml?gid=1980586524&single=true'
    tasks_data = load_data_from_url(tasks_url)
    flow_data = load_data_from_url(flow_url)

    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4", temperature=.2, openai_api_key=openai_api_key)
    
    # Load or create vectorstore
    
    # Load or create vectorstore
    vectorstore_path = './chroma_db'
    embedding_function = OpenAIEmbeddings(api_key=openai_api_key)
    if os.path.exists(vectorstore_path) and os.listdir(vectorstore_path):
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_function)
    else:
        # Convert the tasks to Document objects
        documents = [Document(page_content=task['Journey Name (N)']) for task in tasks_data]
        # Create an embeddings model
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        # Create a Chroma vectorstore from the documents
    
        # Save the vectorstore for future use
        # Save the vectorstore to disk with persistence
        vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
        # vectorstore.save(vectorstore_path)

    # Perform a similarity search
    similar_docs = vectorstore.similarity_search(recipe_name, k=1)
    if similar_docs:
        closest_task = similar_docs[0].page_content
        similarity = np.linalg.norm(np.array(vectorstore.embeddings.embed_query(recipe_name)) - np.array(vectorstore.embeddings.embed_query(closest_task)))
        
        # Get agenda items and methods for the closest task
        agenda_items = get_agenda_items(closest_task, flow_data)
        methods_str = get_methods(closest_task, flow_data)  # New line to get methods
        method_details = get_method_details(methods_str, tasks_data)  # Get method details
        
        if agenda_items and method_details:
            # Create a chain that uses the language model to generate a complete sentence
            template = "Based on your input, I suggest you to follow these steps: {agenda_items}. This suggestion is based on the recipe '{recipe_name}', which is {similarity}% similar to your input. The original recipe that it is matching with is '{closest_task}'."
            prompt = PromptTemplate(template=template, input_variables=["agenda_items", "recipe_name", "similarity", "closest_task"])
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            response = llm_chain.run({"agenda_items": ', '.join(agenda_items), "recipe_name": recipe_name, "similarity": round(similarity * 100, 2), "closest_task": closest_task})
            return jsonify({
                "response": response,
                "details": {
                    "Closest Luma Task": closest_task,
                    "Methods": '| '.join([detail['method'] for detail in method_details]),
                    "Method Details": method_details,
                    "Similarity": f"{similarity}% similar to that task"
                }
            })
        else:
            return jsonify({"error": "Agenda Items or Methods not found for the task"}), 404
    else:
        return jsonify({"error": "Task not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)

