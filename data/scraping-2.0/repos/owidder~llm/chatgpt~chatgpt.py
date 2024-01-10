from llama_index import SimpleDirectoryReader, GPTListIndex, LLMPredictor, ServiceContext
from langchain import OpenAI
import os
from datetime import datetime


def construct_index(directory_path):
    num_outputs = 512
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    docs = SimpleDirectoryReader(directory_path).load_data()
    index = GPTListIndex.from_documents(docs, service_context=service_context)

    return index


index = construct_index("docs")
query_engine = index.as_query_engine()
print(datetime.now())
response = query_engine.query("Please answer the following question as short as possible: What is a knotterbex policy?")
print(response)
print(datetime.now())
response = query_engine.query("Please answer the following question as short as possible: Is it efficient?")
print(response)
print(datetime.now())
