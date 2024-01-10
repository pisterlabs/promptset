import asyncio
import os

import pinecone
from flask import Blueprint, jsonify, request
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from config import pinecone_api_key as pinecone_api_key
from config import pinecone_env as pinecone_env
from flask_login import login_required


similarity = Blueprint("similarity", __name__)

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env, index="proposals")
embeddings = OpenAIEmbeddings()
docsearch = Pinecone.from_existing_index(index_name="proposals", embedding=embeddings)
chat = ChatOpenAI(verbose=True, temperature=0)


@similarity.route("/similarity", methods=["POST"])
@login_required
def similarity_function():
    text_to_search = request.form.get("section-prompt")
    print(request.form)
    prompt = f"""From the context I'm providing, find solutions that can fit into this {text_to_search} . Don't invent, just say 'No related items found' if you don't find anything"""
    prompts = [
        {"title": "similaritySection", "prompt": prompt},
    ]
    qa = RetrievalQA.from_chain_type(
        llm=chat, chain_type="stuff", retriever=docsearch.as_retriever()
    )
    result = asyncio.run(search_similarity(prompts[0], qa))

    print(jsonify(result))
    return jsonify(result), 200


async def search_similarity(prompt_obj, qa):  # receive qa as an argument
    try:
        # Let's run the blocking function in a separate thread using asyncio.to_thread
        response = await asyncio.to_thread(qa.run, prompt_obj["prompt"])
    except Exception as e:
        response = None
    finally:
        return prompt_obj["title"], {
            "question": prompt_obj["prompt"],
            "response": response,
        }
