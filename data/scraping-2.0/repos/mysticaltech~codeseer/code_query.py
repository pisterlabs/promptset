from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import langchain
import os
from rich import print as rprint
import sys

langchain.debug = True

# Load environment variables from .env file
load_dotenv()

# Instantiate the EmbeddingFunction
ef = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

db = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    persist_directory="db",
    embedding_function=ef,
)

llm = ChatOpenAI(
    temperature=0.0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)


query = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "How to modify the max_tokens parameter in the continue codebase?"
)

results = chain(
    {
        "query": query,
    }
)


rprint(f"[bold cyan]Query:[/bold cyan] {results['query']}")
rprint(f"[bold cyan]Result:[/bold cyan] {results['result']}")
