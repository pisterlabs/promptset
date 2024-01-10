from fastapi import FastAPI, Path, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.responses import JSONResponse
from fastapi import HTTPException
import openai
import ast
import json




app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the json file into a Pandas DataFrame
df = pd.read_json('final_embeddings_2.json', orient='records')

@app.get("/")
def home():
    return "<div>Hello World</div>"

print(type(df['ada_embedding'][0]))

@app.post('/get_similar_texts')
async def get_similar_texts(text:str, k: int = 5):
    try:
        openai.api_key = "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"
        response = openai.Embedding.create(input=text,model="text-embedding-ada-002")
        vector = response['data'][0]['embedding']
    except:
        raise HTTPException(status_code=500, detail="openai error")
        
    try:
        # Normalize the input vector
        vector = np.array(vector).reshape(1, -1)
        vector /= np.linalg.norm(vector)

        # Compute cosine similarity between input vector and all embeddings in the DataFrame
        embeddings = np.array(df['ada_embedding'].tolist())
        similarities = cosine_similarity(vector, embeddings)

        # Get the indices of the top K similar texts
        top_indices = np.argsort(similarities[0])[::-1][:k]

        # Get the top K similar texts and their corresponding cosine similarities
        results = []
        for index in top_indices:
            text = df.loc[index, 'text']
            similarity = similarities[0][index]
            results.append({'text': text, 'similarity': similarity})

        return JSONResponse(content=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
