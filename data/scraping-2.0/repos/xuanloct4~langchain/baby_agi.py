

import environment

import os
from collections import deque
from typing import Dict, List, Optional, Any

from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.experimental import BabyAGI

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

import faiss

from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

# Define your embedding model
# embedding = OpenAIEmbeddings()
# Initialize the vectorstore as empty

# embedding_size = 1536   #For chatgpt OpenAI
embedding_size = 768      #For HuggingFace
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embedding.embed_query, index, InMemoryDocstore({}), {})

OBJECTIVE = "Write a weather report for SF today"

# Logging of LLMChains
verbose = False
# If None, will keep on going forever
max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
)

baby_agi({"objective": OBJECTIVE})