from typing import Any, List, Dict
import openai
import logging
from datastore.factory import get_datastore
from models.models import Query, Document, DocumentMetadata, DocumentMetadataFilter
import ntpath
import asyncio

async def startup():
    global datastore
    datastore = await get_datastore()

async def query_database(query_prompt: str, document_id: str) -> Dict[str, Any]:
    """
    Query vector database to retrieve chunk with user's input questions.
    """
    query = Query(
        query=query_prompt,
        filter= DocumentMetadataFilter(
            document_id=document_id
        ),
        top_k=3,
    )
    
    queries = []
    queries.append(query)

    results = await datastore.query(
        queries,
    )

    return results

async def upsert(id: str, content: str):
    """
    Upload one piece of text to the database.
    """
    await startup()
    documents = []
    documents.append(Document(id=id, text=content))

    response_ids = await datastore.upsert(documents)
    return response_ids

async def upsert_file(file_path: str):
    documents = []
    with open(file_path, "rb") as f:
        file_content = f.read()
        documents.append(Document(id=ntpath.basename(file_path), text=file_content))

    response_ids = await datastore.upsert(documents)
    return response_ids

async def delete(id: str, delete_all: bool):
    ids = []
    ids.append(id)
    success = await datastore.delete(
        ids=ids,
        delete_all=delete_all
    )
    return success

def apply_prompt_template(question: str) -> str:
    """
        A helper function that applies additional template on user's question.
        Prompt engineering could be done here to improve the result. Here I will just use a minimal example.
    """
    prompt = f"""
        By considering above input from me, answer the question: {question}
    """
    return prompt


def call_chatgpt_api(user_question: str, chunks: List[str]) -> Dict[str, Any]:
    """
    Call chatgpt api with user's question and retrieved chunks.
    """
    # Send a request to the GPT-3 API
    messages = list(
        map(lambda chunk: {
            "role": "user",
            "content": chunk
        }, chunks))
    question = apply_prompt_template(user_question)
    messages.append({"role": "user", "content": question})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=200,
        temperature=0.7,  # High temperature leads to a more creative response.
    )
    return response


async def ask(user_question: str, document_id: str) -> Dict[str, Any]:
    """
    Handle user's questions.
    """
    await startup()
    # Get chunks from database.
    chunks_response = await query_database(user_question, document_id)
    chunks = []
    for result in chunks_response:
        for inner_result in result.results:
            chunks.append(inner_result.text)
    
    logging.info("User's questions: %s", user_question)
    logging.info("Retrieved chunks: %s", chunks)
    
    response = call_chatgpt_api(user_question, chunks)
    logging.info("Response: %s", response)
    
    return response["choices"][0]["message"]["content"]

