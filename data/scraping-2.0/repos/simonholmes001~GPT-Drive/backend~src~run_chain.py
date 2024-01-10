# export PYTHONPATH=.
import os
from dotenv import load_dotenv
import yaml

import torch
from torch import cuda

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chains import ConversationalRetrievalChain

from backend.set_up import chain_setup, mosaicml_setup, prompt_setup, retriever_setup

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set-up parameters
with open("./backend/set_up/config.yaml","r") as file_object:
    data=yaml.load(file_object,Loader=yaml.SafeLoader)

model_type = data["mosaicml"]["model_type"]
task_type = data["mosaicml"]["task_types"]

device = data["device"]
model = data["chain_parameters"]["model"][2] # 0 mosaicml, 1 openai, 2 ChatOpenAI
openai_model = data["OpenAI"]["gpt3_models"][2]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain_type = data["chain_parameters"]["chain_types"][0]
embedding_model_name = data["embedding"][0] # 0 openai, 1 hugging
vectorDB = data["vectorDB"][0] # 0 chroma, 1 FAISS
data_source = data["data_source"][0] # 0 faurecia, 1 basler, 2 autosar
search_kwargs= data["search_kwargs"][1]
search_type= data["search_type"][1]

if __name__ == "__main__":
    # Set-up model choices
    if model == "mosaicml-30B":
        mosaicml = mosaicml_setup.MosaicmlSetUp(device=device, model_type=model_type, task_type=task_type)
        model_setup = mosaicml.model_setup(device=device, model_type=model_type)
        stopping_criteria = mosaicml.stopping_criteria()
        llm = mosaicml.generate_text(task_type=task_type, stopping_criteria=stopping_criteria)
    if model == "openai":
        llm = OpenAI(temperature=0, streaming=True, callbacks=[StreamingStdOutCallbackHandler()], model_name=openai_model, verbose=True)
    if model == "ChatOpenAI":
        llm = ChatOpenAI(temperature=0, streaming=True, callbacks=[StreamingStdOutCallbackHandler()], model_name=openai_model, verbose=True,)

    # Prompt set-up
    prompt = prompt_setup.Prompt()
    query = prompt.set_query()
    set_template = prompt.set_prompt_template()
    prompt_template = prompt.set_template_structure(set_template)
    
    # Chain set-up
    langchain = chain_setup.Chain(llm=llm, chain_type=chain_type, prompt=prompt_template)
    chain = langchain.never_break_the_langchain(llm=llm, chain_type=chain_type, prompt=prompt_template)

    # Retriever set-up
    if embedding_model_name == "openai": 
        embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    if embedding_model_name == "hugging":
        embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    retriever = retriever_setup.Retriever(data_source=data_source, embedding_model=embedding, embedding_model_name=embedding_model_name, 
    search_kwargs=search_kwargs, search_type=search_type, query=query)
    if vectorDB == "chroma":
        docs = retriever.chroma(data_source=data_source, embedding=embedding, embedding_model_name=embedding_model_name, 
        search_kwargs=search_kwargs, search_type=search_type, query=query)
    if vectorDB == "faiss":
        docs = retriever.faiss(data_source=data_source, embedding=embedding, embedding_model_name=embedding_model_name, 
        search_kwargs=search_kwargs, search_type=search_type, query=query)

    print("\n")
    print("--- START ---")
    print("\n")
    print("Listing all query parameters...")
    print(f"Model type: {model}")
    if model == "openai" or model == "ChatOpenAI":
        print(f"OpenAI model type: {openai_model}")
    print(f"VectorDB: {vectorDB}")
    print(f"Embedding model: {embedding_model_name}")
    print(f"Query: {query}")
    print("\n")
    chain.run(input_documents=docs, question=query, return_only_outputs=True, memory=memory, verbose=True)
    # qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=docs, memory=memory)
    # result = qa({"question": query})
    # print(result["answer"])
    print("\n")
    print("--- FIN ---")
    print("\n")