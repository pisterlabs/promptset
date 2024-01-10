import json
import langchain
import json
import openai
import os
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, OllamaEmbeddings, HuggingFaceEmbeddings
from langchain.schema import Document

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.chat_models import ChatOpenAI

import yaml
import typing
model_map = """
llm:
  gpt-3.5-turbo : openAI
  llama2 : ollama

embedding:
  text-embedding-ada-002 : openAI
  llama2 : ollama
  all-MiniLM-L6-v2 : sent""" #! AUGMENT THIS WITH OTHER MODELS
api_key = "none"

model_map = yaml.load(model_map, Loader=yaml.FullLoader)
print(model_map)

def get_embedding_model(modelname : str, model_map : typing.Dict = model_map):
    try:
        model_type = model_map['embedding'][modelname].lower()
        if model_type == 'openai':
            return OpenAIEmbeddings(model=modelname, api_key=api_key)
        elif model_type == 'ollama':
            return OllamaEmbeddings(model=modelname)
        elif model_type == 'sent':
            return HuggingFaceEmbeddings(model_name=modelname)
        else:
            return KeyError(f"Model Type {model_type} not found")
    except KeyError:
        print(KeyError("Model not found"))
        return None


def get_retriever(docs : typing.List, modelname : str, model_map : typing.Dict = model_map):
    embedding_model = get_embedding_model(modelname)
    assert embedding_model is not None
    Descs = [
                Document(page_content=t['description'], metadata={"index": i})
                for i, t in enumerate(docs)
    ] #! Change this according to your needs

    Descs_Store = FAISS.from_documents(Descs, embedding_model)
    Retriever = Descs_Store.as_retriever()
    return Retriever


def get_tools(query : str, documentation : typing.List, retriever : typing.Any):
    docs = retriever.get_relevant_documents(query)
    tools = [documentation[d.metadata["index"]]['tool'] for d in docs]
    arguments = [documentation[d.metadata["index"]] for d in docs]
    return tools, arguments