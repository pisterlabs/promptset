# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from semantic_kernel.text import text_chunker
from langchain.document_loaders import WebBaseLoader
from typing import Tuple
from PyPDF2 import PdfReader
from langchain.document_loaders import YoutubeLoader
from langchain.utilities import WikipediaAPIWrapper
from langchain.document_loaders import Docx2txtLoader
import semantic_kernel as sk
import pickle
import os
from typing import List, Dict, Union, Any
import asyncio
from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextCompletion, HuggingFaceTextEmbedding
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.orchestration.context_variables import ContextVariables
import openai

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "http://localhost:5173"}})


async def setKernelForData(text, model, store_name) -> None:

    kernel = sk.Kernel()

    api_key, org_id = sk.openai_settings_from_dot_env()

    if model == "Hugging Face":
        kernel = sk.Kernel()
        print("Setting up Hugging Face...")
        kernel.add_text_completion_service(
            "google/flan-t5-xl", HuggingFaceTextCompletion("google/flan-t5-xl")
        )
        kernel.add_text_embedding_generation_service(
            "sentence-transformers/all-mpnet-base-v2", HuggingFaceTextEmbedding(
                "sentence-transformers/all-mpnet-base-v2")
        )
    elif model == "OpenAI":
        kernel = sk.Kernel()
        print("Setting up OpenAI API key...")
        kernel.add_text_completion_service(
            "dv", sk_oai.OpenAITextCompletion(
                "text-davinci-003", api_key, org_id)
        )
        kernel.add_text_embedding_generation_service(
            "ada", sk_oai.OpenAITextEmbedding(
                "text-embedding-ada-002", api_key, org_id)
        )

    # creating chunks
    chunks = text_chunker._split_text_lines(text, 1000, False)
    if not os.path.exists(f"{store_name}. pkl"):
        with open(f"{store_name}. pkl", "wb") as f:
            pickle.dump(chunks, f)
        print("Embeddings Computation Completed")


@app.route('/setData', methods=['POST'])
def chat_route():
    data = request.get_json()
    # Extract data from the request
    pdf_file = data.get('pdf_file')
    youtube_url = data.get('youtube_url')
    web_url = data.get('web_url')
    model = data.get('model')

    # Process the data and get the chat response
    text = ""
    if pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    if youtube_url:
        loader = YoutubeLoader.from_youtube_url(
            youtube_url, add_video_info=True)
        result = loader.load()
        k = str(result[0])
        text += "This is youtube URL" + k
    if web_url:
        store_name = web_url.split("//")[-1].split("/")[0]
        if not os.path.exists(f"{store_name}.pkl"):
            r = requests.get(web_url)
            soup = BeautifulSoup(r.text, "lxml")
            links = list(set([a["href"]
                         for a in soup.find_all("a", href=True)]))
            k = ""
            links.remove('https://carepvtltd.com/shifa-care/')
            links.remove('https://carepvtltd.com/case-university/')
            for link in links:
                if link.startswith('http://carepvt') or link.startswith('https://carepvt'):
                    print("Checking for", link)
                    loader = WebBaseLoader(link)
                    data = loader.load()
                    k += str(data[0])
            text += "This is website URL" + k

    asyncio.run(setKernelForData(text, model, store_name))

    # Return the chat response as a JSON object
    return jsonify({"response": "Data Recorded Successfully"})


if __name__ == '__main__':
    app.run(debug=True)
