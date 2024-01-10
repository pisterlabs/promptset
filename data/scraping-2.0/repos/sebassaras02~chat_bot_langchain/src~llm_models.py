from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from embeddings import get_vector_db
from utils import get_query_from_user, load_llm_model, process_query_with_memory, memory_object
from langchain import OpenAI
import os
from rich.console import Console

console = Console()

# define a function to run the model
def run_chat():
    # load vectorial database
    db = get_vector_db('../data/historia.txt', 'e-sas_company')
    retriever_chroma = db.as_retriever(search_kwargs={'k': 5})
    llm = load_llm_model('../openaikey.txt', 300)
    k = 0
    memory = memory_object('../openaikey.txt')
    while True:
        if k == 0:
            console.print("Chat running. Type 'exit' to quit.", style="blue")
        query = get_query_from_user()
        if query == "exit":
            break
        response = process_query_with_memory(llm, retriever_chroma, query, memory)
        k += 1
        return response
    
    
