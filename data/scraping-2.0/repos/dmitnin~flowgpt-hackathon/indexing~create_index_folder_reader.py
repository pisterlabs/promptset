#!/usr/bin/env python

from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import sys
import os
import openai


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

openai.api_key = OPENAI_API_KEY;


def construct_index(directory_path):
    max_input_size = 4096 * 10
    num_output = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600
    chunk_overlap_ratio = 0.1

    prompt_helper = PromptHelper(
        max_input_size=max_input_size,
        num_output=num_output,
        max_chunk_overlap=max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
        chunk_overlap_ratio=chunk_overlap_ratio)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_output))

    documents = SimpleDirectoryReader(input_dir=directory_path, recursive=True).load_data()

    index = GPTVectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.storage_context.persist(persist_dir="./index")

    return index


index = construct_index("../docs")
