import re
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64
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
import semantic_kernel as sk2
import pickle
import os
from typing import List, Dict, Union, Any
import asyncio
from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextCompletion, HuggingFaceTextEmbedding
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAITextEmbedding, OpenAITextCompletion
import openai
import time

app = Flask(__name__)
CORS(app, origins="http://localhost:5173")


async def import_information_from_vector_store(store_name) -> List[Dict[str, Union[str, Any]]]:
    # Load information from the vector store pickle file
    with open("carepvtltd.com. pkl", "rb") as file:
        vector_store_info = pickle.load(file)

    # Ensure vector_store_info is a list of dictionaries
    if not isinstance(vector_store_info, list):
        vector_store_info = [vector_store_info]

    return vector_store_info


async def populate_memory(kernel: sk.Kernel, store_name, model) -> None:
    # Import information from vector store
    vector_store_info = await import_information_from_vector_store(store_name)
    # print("this is the vector store info", vector_store_info)

    # Add imported documents to the semantic memory
    count = 1
    if model == "Hugging Face":
        for info in vector_store_info:
            count += 1
            # Save the information to the semantic memory
            await kernel.memory.save_information_async(
                "aboutMe", id="info"+str(count), text=info
            )
            print("Populating for info", count)
        # await kernel.memory.save_information_async(
        #     "aboutMe", id="info"+str(count), text="""Center for Advanced Research in Engineering [CARE]
        #     Pvt Ltd is one of the most celebrated Information and Communication Technology (ICT) organizations in Pakistan. While successfully executing projects with innovative designs, CARE has also claimed regional distinction by consecutively winning
        #     13 Pakistan Software House Association (P@SHA) and 12 Asia Pacific ICT Alliance Awards (APICTA) since 2010.
        #     Who We Are A group of Professionals comprising several PhDs, Masters and Graduates from renowned academic institutions, having polished their technology skills by working in highly reputed industries.
        #     About Care Vision Mission To bring Pakistan on World Technology map by crafting novel information and communication technology solutions, with innovation and creativity, to solve customers’ problems and focusing on solutions and technologies that have critical local need with immense global scope and then taking the local success to new customers across the globe.
        #     """
        # )
    else:
        await kernel.memory.save_information_async(
            "aboutMe", id="info", text="""
            About Us:
            Where your imagination sparks into reality through the power of AI. Dive into the realm of endless possibilities, crafting unique, blockchain-enshrined artworks with just a few clicks. 
            Artigenious is a platform that allows you to create unique NFTs using the power of AI.
            How it works:
            Step 1
            Start by entering a creative prompt that describes the NFT you want to generate. This could be anything from an abstract concept to 
            a detailed description of an image.
            Step 2
            After entering your prompt, customize various layers to add uniqueness to your NFT. This could include adjusting colors, patterns, or adding specific elements.
            Step 3
            Once you're satisfied with the customization, click 'Generate' to create your NFT collection. Each piece will be unique and based on the specifications of your prompt and customizations.
            """
        )
        # for info in vector_store_info:
        #     count += 1
        #     # Save the information to the semantic memory
        #     await kernel.memory.save_information_async(
        #         "aboutMe", id="info"+str(count), text=info
        #     )
        #     print("Populating for info", count)
        #     time.sleep(25)


async def search_memory_examples(kernel: sk.Kernel, query) -> None:
    result = await kernel.memory.search_async("aboutMe", query, limit=1)
    return result


async def setup_chat_with_memory(
    kernel: sk.Kernel,
) -> Tuple[sk.SKFunctionBase, sk.SKContext]:
    sk_prompt = """
    You are a friendly user chatbot. You are having a conversation with a user. The user is asking you questions about me. You are answering the user's questions. You are trying to be as helpful as possible. You are trying to be as friendly as possible. You are trying to be as polite as possible. You are trying to be as informative as possible. You are trying to be as accurate as possible. You are trying to be as helpful as possible. You are trying to be as friendly as possible. You are trying to be as polite as possible. You are trying to be as informative as possible. You are trying to be as accurate as possible. You are trying to be as helpful as possible. You are trying to be as friendly as possible. You are trying to be as polite as possible. You are trying to be as informative as possible. You are trying to be as accurate as possible. You are trying to be as helpful as possible. You are trying to be as friendly as possible. You are trying to be as polite as possible. You are trying to be as informative as possible. You are trying to be as accurate as possible.
    Chatbot should only answer from the given facts only.
    It should say 'I apologize, but it appears that the information you're requesting is beyond my current knowledge.
                 As an AI language model, my training only goes up to given data, and I am not aware of any events or developments that occurred outside it. Is there anything else I can assist you with that falls within my knowledge range?' if
    the answer is not present in the given facts.
    Information about me, from previous conversations:
    - {{$fact1}} {{recall $fact1}}
    Chat:
    {{$chat_history}}
    User: {{$user_input}}
    ChatBot:""".strip()

    chat_func = kernel.create_semantic_function(
        sk_prompt, max_tokens=200, temperature=0.8
    )

    context = kernel.create_new_context()

    context[sk.core_skills.TextMemorySkill.COLLECTION_PARAM] = "aboutMe"
    context[sk.core_skills.TextMemorySkill.RELEVANCE_PARAM] = 1

    context["chat_history"] = ""

    return chat_func, context


def clean_data(text):
    # Remove newlines, multiple spaces, and tabs
    cleaned_text = re.sub(r'\n+|\t+', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    cleaned_text = cleaned_text.replace('\\n', ' ')
    cleaned_text = cleaned_text.replace('\\t', ' ')
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text


async def chat(
    kernel: sk.Kernel, chat_func: sk.SKFunctionBase, context: sk.SKContext, model, query
) -> None:

    context["user_input"] = query

    result = await search_memory_examples(kernel, query)

    if model == "Hugging Face":
        chunk_size = 500
        remaining_text = str(result[0].text)
        context["chat_history"] = ""  # Initialize chat history
        count = 0
        while remaining_text:

            # Get the next chunk of text
            current_chunk = remaining_text[:chunk_size]
            remaining_text = remaining_text[chunk_size:]

            # Replace "\t" characters and update the context with the current chunk
            c = current_chunk.replace("\\t", "")
            stripped_text = ""
            for i in range(len(c)):
                stripped_text += c[i]
            context["fact1"] = stripped_text

            # Call the kernel with the current context
            answer = await kernel.run_async(chat_func, input_vars=context.variables)
            context["chat_history"] += f"\nUser:> {query}\nChatBot:> {answer}\n"

            response_without_answer = "I apologize, but it appears that the information you're requesting is beyond my current knowledge. As an AI language model, my training only goes up to given data, and I am not aware of any events or developments that occurred outside it. Is there anything else I can assist you with that falls within my knowledge range?"
            count += 1
            print(stripped_text)
            if response_without_answer in str(answer):
                # If "I don't know" is in the response, break the loop
                continue
            else:
                break
            # Return the final chat history as a JSON object
        return answer

    else:
        context["fact1"] = result[0].text
        if query != "":
            answer = await kernel.run_async(chat_func, input_vars=context.variables)
            context["chat_history"] += f"\nUser:> {query}\nChatBot:> {answer}\n"
            return answer  # Return the chat response as a JSON object


async def setKernel(query, model, store_name) -> None:

    api_key, org_id = sk.openai_settings_from_dot_env()

    if model == "Hugging Face":
        kernel2 = sk2.Kernel()
        print("Setting up Hugging Face...")
        kernel2.add_text_completion_service(
            "google/flan-t5-large", HuggingFaceTextCompletion(
                "google/flan-t5-large")
        )

        kernel2.add_text_embedding_generation_service(
            "sentence-transformers/all-mpnet-base-v2", HuggingFaceTextEmbedding(
                "sentence-transformers/all-mpnet-base-v2")
        )
        kernel2.register_memory_store(
            memory_store=sk2.memory.VolatileMemoryStore())
        kernel2.import_skill(sk2.core_skills.TextMemorySkill())

        print("Populating memory...")
        await populate_memory(kernel2, store_name, model)

        # print("Asking questions... (manually)")
        # await search_memory_examples(kernel)

        print("Setting up a chat (with memory!)")
        chat_func, context = await setup_chat_with_memory(kernel2)

        print("Begin chatting (type 'exit' to exit):\n")

        return await chat(kernel2, chat_func, context, model, query)

    elif model == "OpenAI":
        kernel = sk.Kernel()
        print("Setting up OpenAI API key...")
        kernel.add_chat_service("chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id))
        print("Setting up OpenAI text completion...")
        kernel.add_text_embedding_generation_service("ada", OpenAITextEmbedding("text-embedding-ada-002", api_key, org_id))
        print("adding memory store...")
        kernel.register_memory_store(
            memory_store=sk.memory.VolatileMemoryStore())
        print("importing skill...")
        kernel.import_skill(sk.core_skills.TextMemorySkill())

        print("Populating memory...")
        await populate_memory(kernel, store_name, model)

        # print("Asking questions... (manually)")
        # await search_memory_examples(kernel)

        print("Setting up a chat (with memory!)")
        chat_func, context = await setup_chat_with_memory(kernel)

        print("Begin chatting (type 'exit' to exit):\n")
        return await chat(kernel, chat_func, context, model, query)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})


@app.route('/chats', methods=['POST'])
def chats():
    query = request.json.get('query', '').strip()
    model = request.json.get('model', '').strip()
    print(query)
    store_name = "https://carepvtltd.com/#".split("//")[-1].split("/")[0]
    answer = asyncio.run(setKernel(query, model, store_name))
    result = {
        'question': query,
        'answer': str(answer),
        'source_documents': []
    }
    return jsonify(result)


async def setDataForKernel(text, model, store_name) -> None:
    # creating chunks
    chunks = text_chunker._split_text_lines(text, 3500, False)
    if not os.path.exists(f"{store_name}. pkl"):
        with open(f"{store_name}. pkl", "wb") as f:
            pickle.dump(chunks, f)
        print("Embeddings Computation Completed")


@app.route('/setData', methods=['POST', 'OPTIONS'])
def setData():
    print(request.method)
    if request.method == 'OPTIONS':
        # Handle the preflight CORS request
        response_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
        }
        # Return empty response with appropriate CORS headers
        return ('', 204, response_headers)
    else:
        print(request.form)
        print(request.files)
        # Access the uploaded PDF file
        pdf_file = request.files.get('pdfFile')

        # Access the form fields
        youtube_url = request.form.get('ytURL')
        web_url = request.form.get('webURL')
        model = request.form.get('model')

        print(web_url)

        # Process the data and get the chat response
        text = ""
        store_name = ""
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

        text = clean_data(text)
        asyncio.run(setDataForKernel(text, model, store_name))

        # Return the chat response as a JSON object
        return jsonify({"response": "Data Recorded Successfully"})


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5002, debug=True)
