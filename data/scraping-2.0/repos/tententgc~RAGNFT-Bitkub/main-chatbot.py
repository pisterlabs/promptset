import numpy as np
import openai
import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
import glob
import pandas as pd
import re
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken
import base64
import time
from fastapi import FastAPI, HTTPException
import uvicorn

from pydantic import BaseModel
import dotenv

app = FastAPI()



openai.api_key = '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' 

def predict(message, history):
    print(message)
    def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/jpeg;base64,{encoded_string}"


    def normalize_text(s, sep_token = " \n "):
        s = re.sub(r'\s+',  ' ', s).strip()
        s = re.sub(r". ,","",s)
        # remove all instances of multiple spaces
        s = s.replace("..",".")
        s = s.replace(". .",".")
        s = s.replace("\n", "")
        s = s.strip()
        return s

    def sim_text(input_text):
        pdf_paths = glob.glob('./data/*.pdf')

        df = pd.DataFrame(columns=['text'])

        for path in pdf_paths:
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings(openai_api_key=openai.api_key))
            docs = faiss_index.similarity_search(input_text, k=5)
            for doc in docs:
                df.loc[len(df.index)] = doc.page_content
        df['text']= df["text"].apply(lambda x : normalize_text(x))
        tokenizer = tiktoken.get_encoding("cl100k_base")

        df['n_tokens'] = df["text"].apply(lambda x: len(tokenizer.encode(x)))
        df = df[df.n_tokens<8192]

        embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

        df['ada_v2'] = df["text"].apply(lambda x : embeddings.embed_query(x))

        embedding = get_embedding(
                input_text,
                engine="text-embedding-ada-002" # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
            )

        df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))
        res = (
                df.sort_values("similarities", ascending=False)
                .head(3)
            )

        top5_text = " \n ".join(res.text[:1].values)

        return top5_text

    start_time = time.time()
    history_openai_format = []
    if len(history)>0:
      for human, assistant in history[-1:]:
          history_openai_format.append({"role": "user", "content": human })
          history_openai_format.append({"role": "assistant", "content":assistant[int(len(assistant)*0.9):]})
    history_openai_format.append({"role": "system", "content":"คุณคือผู้ช่วยตอบคำถามเกี่ยวกับ NFT บน Chain ของเรา คุณสามารถถามคำถามเกี่ยวกับ NFT ได้เลยครับ ตอบเป็นไทยด้วย"})
    history_openai_format.append({"role": "assistant", "content": sim_text(message)})
    history_openai_format.append({"role": "user", "content":message})
    end_time = time.time()
    execution_time = start_time - end_time
    print("history Execution time: ",execution_time)

    start_time = time.time()
    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages= history_openai_format,
        temperature=0.1
    )
    end_time = time.time()
    execution_time = start_time - end_time
    print(response['choices'][0]["message"]["content"])
    return response['choices'][0]["message"]["content"]

answer = predict("NFT คืออะไร",[])
print("answer: ",answer)