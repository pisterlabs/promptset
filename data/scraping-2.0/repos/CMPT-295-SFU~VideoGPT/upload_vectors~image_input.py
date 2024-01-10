#%% 

from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
import base64
import requests
import os
import langchain
import json
from openai import OpenAI
import numpy as np
from itertools import islice
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type
from tokenizers import Tokenizer
import tiktoken
# from tokenizers.models import BPE
# from tokenizers.pre_tokenizers import Whitespace
import openai
from langchain.embeddings import OpenAIEmbeddings
import pinecone
import glob
from tqdm import tqdm
import fire

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def weeklypng_to_vectors(path):
    print("Hello")
# OpenAI API Key
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    chat = ChatOpenAI(model="gpt-4-vision-preview",
                      max_tokens=4096, openai_api_key=api_key)
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key, model="text-embedding-ada-002")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pinecone.init(api_key=pinecone_key, environment="us-west1-gcp-free"
                  )
    # Initialize the index
    index = pinecone.Index(index_name="295-youtube-index")

    # index.delete(delete_all=True, namespace="Slides")

    # Get all the directories in the input directory
    print(path)
    directories = glob.glob(path)

    for directory_path in directories:
        # Use glob to match the pattern '/*.png'
        files = glob.glob(directory_path + "*.png")
        # Get parent name
        parent_directory_name = os.path.basename(
            os.path.dirname(os.path.dirname(directory_path)))
        # print(parent_directory_name)
        # Get directory name
        directory_name = os.path.basename(os.path.dirname(directory_path))
        # print(directory_name)
        path = os.path.join(parent_directory_name, directory_name) + ".pdf"
        metadatas = []
        vectors = []
        ids = []
        print(path)
        for image_path in tqdm(files):
            # Get path to pdf
            # Getting the base64 string
            base64_image = encode_image(image_path)

            content_json = chat.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text",
                                "text": "What’s in this image? Return the  result as json with the following fields: title (string of 2-3 sentences), description (string))"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "auto",
                            },
                        },
                    ]
                )
            ])
            content_string = content_json.content.strip('```json\n')
            content_dict = json.loads(content_string)

            content_dict["metadata"] = {"file": "", "slide": "",
                                    "description": "#######CONTEXT#####\n" + content_dict["description"]}
            # Tokenize text and create embedding vector using openai
            text = content_dict["title"] + " " + content_dict["description"]
            query_result = embeddings.embed_query(text)
            slide_number = os.path.splitext(os.path.basename(image_path))[
                0].replace("Slide", "")
            metadata = {"file": path, "Slide": slide_number, "description": text}
            ids.append(path+"#"+slide_number)
            metadatas.append(metadata)
            vectors.append(query_result)

    index.upsert(zip(ids, vectors, metadatas), namespace="Slides")


    # Upsert data
    # index.upsert([(path, query_result, metadata)], namespace="Slides")
if __name__ == '__main__':
      fire.Fire(weeklypng_to_vectors)
