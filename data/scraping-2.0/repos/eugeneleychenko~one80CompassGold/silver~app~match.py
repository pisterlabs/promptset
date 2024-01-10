from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain.vectorstores import FAISS
from fastapi.middleware.cors import CORSMiddleware
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
import requests
import os
from dotenv import load_dotenv
import uvicorn

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


@app.post("/find_closest_match")
async def find_closest_match(payload: dict):
    user_input = payload.get("user_input")
    if not user_input:
        raise HTTPException(status_code=400, detail="User input is required in the payload")

    # Load the JSON from the provided URL
    json_url = "https://opensheet.elk.sh/1vgJJHgyIrjip-6Z-yCW6caMgrZpJo1-waucKqvfg1HI/5"
    data = load_data_from_url(json_url)
    #   
    agenda_items = [item.get('Journey Name (N)', 'Default Description') for item in data]

    # Initialize embeddings and check if FAISS index exists
    embeddings = OpenAIEmbeddings()
    faiss_index = load_faiss_index_if_exists(embeddings)
    if faiss_index is None:
        # Create a new index and save it
        documents = [Document(page_content=item) for item in agenda_items]
        faiss_index = FAISS.from_documents(documents, embeddings)
        save_faiss_index(faiss_index)
    # Find the closest match
    similar_docs = faiss_index.similarity_search(user_input, k=1)
    if similar_docs:
        closest_match = similar_docs[0].page_content
        # Find the entry with the matching Journey Name and gather all associated methods
        methods = []
        alternatives_list = []
        
        # Load methods data from URL
        methods_url = "https://opensheet.elk.sh/1vgJJHgyIrjip-6Z-yCW6caMgrZpJo1-waucKqvfg1HI/6"
        methods_data = load_data_from_url(methods_url)
        # print("methods mapping: ", methods_mapping)
        
        
        # Match 'closest_match' with Journey Name (N) in the json_url
        matched_entries = [entry for entry in data if entry.get('Journey Name (N)') == closest_match]
        if not matched_entries:
            raise HTTPException(status_code=404, detail="No matching agenda items found")

        # Retrieve all the Methods (N) that are associated with Journey Name (N) in the json_url
        methods = []
        for entry in matched_entries:
            entry_methods = entry.get('Methods (N)', '')
            if entry_methods:
                methods.extend(entry_methods.split('; '))

        # Remove duplicates from methods list
        methods = list(set(methods))

        # From the methods_data json, find all the methods in the Unique key
        # and return Alt 1, Alt 2, Alt 3, Description (short), AI Response
        alternatives_list = []
        for method in methods:
            method_entry = next((item for item in methods_data if item['Uniques'] == method), None)
            if method_entry:
                alternatives = {
                    'Alt 1': method_entry.get('Alt 1', ''),
                    'Alt 2': method_entry.get('Alt 2', ''),
                    'Alt 3': method_entry.get('Alt 3', ''),
                    'Description (short)': method_entry.get('Description (short)', ''),
                    'AI Response': method_entry.get('AI Response', ''),
                    'Sidebar Description': method_entry.get('Sidebar Description', ''),
                    'Helpful Hints': method_entry.get('Helpful Hints', ''),
                    'Helpful Resources': method_entry.get('Helpful Resources', ''),
                    'Templates': method_entry.get('Templates', ''),
                    'Examples': method_entry.get('Examples', ''),
                }
                alternatives_list.append({method: alternatives})

        return JSONResponse({
            "closest_match": closest_match,
            "methods": methods,
            "alternatives": alternatives_list
        })
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)




