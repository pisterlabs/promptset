import logging
import sys
import torch
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding

def configure_logging():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def setup_llm():
    system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
    query_wrapper_prompt = SimpleInputPrompt("{query_str}")

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",
        model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
    )

    return llm

def setup_embedding_model():
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )
    return embed_model

def setup_service_context(llm, embed_model):
    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )
    return service_context

def create_index(data_directory, service_context):
    documents = SimpleDirectoryReader(data_directory).load_data()
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    return index

def main():
    configure_logging()

    # Install required packages
    pip install pypdf
    pip install python-dotenv
    pip install -q transformers einops accelerate langchain bitsandbytes
    pip install sentence_transformers
    pip install llama-index

    # Replace with your data directory path
    data_directory = "/content/data/"

    llm = setup_llm()
    embed_model = setup_embedding_model()
    service_context = setup_service_context(llm, embed_model)
    index = create_index(data_directory, service_context)
    query_engine = index.as_query_engine()

    while True:
        query = input("Enter your query: ")
        response = query_engine.query(query)
        print(response)

if __name__ == "__main__":
    main()
