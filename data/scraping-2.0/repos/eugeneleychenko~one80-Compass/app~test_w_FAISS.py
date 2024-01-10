from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import requests
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
import os
import numpy as np
import os.path

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_origins=["http://localhost:3000"],  # Allows only specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

FAISS_INDEX_FILE = 'faiss_index.bin'

def save_faiss_index(db):
    db.save_local(FAISS_INDEX_FILE)

def load_faiss_index_if_exists(embeddings):
    if os.path.exists(FAISS_INDEX_FILE):
        return FAISS.load_local(FAISS_INDEX_FILE, embeddings)
    return None

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


@app.post('/get_recipe')
async def get_recipe(request: Request):
    content = await request.json()
    recipe_name = content.get('recipe_name')
    if not recipe_name:
        raise HTTPException(status_code=400, detail="No recipe name provided")

    tasks_url = 'https://gs.jasonaa.me/?url=https://docs.google.com/spreadsheets/d/e/2PACX-1vSmp889ksBKKVVwpaxhlIzpDzXNOWjnszEXBP7SC5AyoebSIBFuX5qrcwwv6ud4RCYw2t_BZRhGLT0u/pubhtml?gid=1980586524&single=true'
    flow_url = 'https://gs.jasonaa.me/?url=https://docs.google.com/spreadsheets/d/e/2PACX-1vSmp889ksBKKVVwpaxhlIzpDzXNOWjnszEXBP7SC5AyoebSIBFuX5qrcwwv6ud4RCYw2t_BZRhGLT0u/pubhtml?gid=1980586524&single=true'
    tasks_data = load_data_from_url(tasks_url)
    flow_data = load_data_from_url(flow_url)

    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4", temperature=.2, openai_api_key=openai_api_key)
    
    # Create an embeddings model
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    # Replace the FAISS index creation with a check to load if it exists
    # If it doesn't exist, create it and save
    db = load_faiss_index_if_exists(embeddings)
    if db is None:
        documents = [Document(page_content=task['Journey Name (N)']) for task in tasks_data]
        db = FAISS.from_documents(documents, embeddings)
        save_faiss_index(db)

    # Perform a similarity search
    similar_docs = db.similarity_search(recipe_name, k=1)
    if similar_docs:
        closest_task = similar_docs[0].page_content
        similarity = np.linalg.norm(np.array(embeddings.embed_query(recipe_name)) - np.array(embeddings.embed_query(closest_task)))
        
        # Get agenda items and methods for the closest task
        agenda_items = get_agenda_items(closest_task, flow_data)
        methods_str = get_methods(closest_task, flow_data)  # New line to get methods
        method_details = get_method_details(methods_str, tasks_data)  # Get method details
        
        if agenda_items and method_details:
            # Create a chain that uses the language model to generate a complete sentence
            template = "Based on your input, I suggest you to follow these steps: {agenda_items}. This suggestion is based on the recipe '{recipe_name}', which is {similarity}% similar to your input. The original recipe that it is matching with is '{closest_task}'."
            prompt = PromptTemplate(template=template, input_variables=["agenda_items", "recipe_name", "similarity", "closest_task"])
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            response = await llm_chain.run({"agenda_items": ', '.join(agenda_items), "recipe_name": recipe_name, "similarity": round(similarity * 100, 2), "closest_task": closest_task})
            return JSONResponse({
                "response": response,
                "details": {
                    "Closest Luma Task": closest_task,
                    "Methods": '| '.join([detail['method'] for detail in method_details]),
                    "Method Details": method_details,
                    "Similarity": f"{similarity}% similar to that task"
                }
            })
        else:
            raise HTTPException(status_code=404, detail="Agenda Items or Methods not found for the task")
    else:
        raise HTTPException(status_code=404, detail="Task not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

