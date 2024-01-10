from typing import Any, List, Dict

# Chat packages
import torch
import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain import HuggingFaceHub

# Ollama for local machines
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Summarization packages
from langchain.chains.llm import LLMChain
# from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


# Global variables
load_dotenv()   # Load environment variables from .env file
huggingfacehub_api_token = os.getenv("HUGGINGFACE_API_KEY")
mistral_repo = 'mistralai/Mistral-7B-Instruct-v0.1'


# Tokenizer
embedd_model = 'BAAI/bge-reranker-large'
model_kwargs = {"device": 0}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=embedd_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


# Building LLM
llm = Ollama(model="mistral",
             verbose=True,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# Function to call LLM and generate response
def run_llm_summarize():

    map_prompt = hub.pull("rlm/map-prompt")
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
   

    loader = PyPDFLoader("/Users/Vincent/Berkeley/w210/compliance_copilot_prototype/copilot-prototype/backend/uploads/NIST.IR.8270.pdf")
    docs = loader.load()

    reduce_prompt = hub.pull("rlm/map-prompt")

    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )
        
    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0)

    split_docs = text_splitter.split_documents(docs)

    summary = map_reduce_chain.run(split_docs)

    return summary

# Function to call LLM and generate response
def run_llm(vector_database: Any, query: str, chat_history: List[Dict[str, Any]] = []):

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_database.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 50}),
        return_source_documents=True
    )
    
    results = qa({"question": query,  "chat_history": chat_history})
    response = results["answer"] 
    sources = [doc.metadata["page"] for doc in results["source_documents"]]

    return response, sources

