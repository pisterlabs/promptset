from decouple import config, Config, RepositoryEnv # python-decouple
from datetime import datetime
import pandas as pd
import numpy as np
import tiktoken
import pinecone # pinecone-client for vector database
import openai
from pydantic import BaseModel
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity # import scikit-learn
from fastapi import APIRouter, HTTPException

class QA(BaseModel):
    question: str
    answer: str

# Config(RepositoryEnv("../../.env"))
router = APIRouter()

tokenizer = tiktoken.get_encoding("cl100k_base")

index_name = 'football-buzz'
pinecone.init( 
    api_key=str(config('API_PINECONE')),
    environment="us-east-1-aws")
index = pinecone.Index(index_name=index_name)

openai.api_key = str(config('API_OPENAI'))

def create_context(
    question, max_len
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    pinecone_results = index.query([q_embeddings], top_k=25, include_metadata=True, include_Values=True, 
        namespace=index_name)
    
    returns = []
    cur_len = 0
    i = 0
    
    for match in pinecone_results['matches']:
        
        i += 1
        cur_len += len(tokenizer.encode(match['metadata']['text'])) + 4
        if cur_len > max_len:
            break
        returns.append("Example " + str(i) + ":\n" + match['metadata']['text'])

    # Return the context
    return "\n\n".join(returns)

def answer_question(
    question,
    model="text-davinci-003",
    max_tokens=1000,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        max_len=2750
    )
    
    author = str(config('AUTHOR')),
    
    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f'"""\nYou are a funny sports journalist writing an article based on a prompt.  Write in the style of {author} and use two curse words.  Use the context below to answer the question.  Use this format, replacing text in brackets with the result.  Do not inclued the brackets in the output:\n\nArtilce:\n[Introductory paragraph]\n\n# [Name of Topic 1]\n[Paragraph about topic 1]\n\n[Concluding paragraph]\n\nContext:\n\n{context}"""\n\nQuestion: {question}?\n',
            temperature=1,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return e
    
@router.post("/compose/", response_model=QA)
async def compose(qa: QA):
    answer = answer_question(question=qa.question)
    response = QA(question=qa.question, answer=answer)
    print(response)
    return response