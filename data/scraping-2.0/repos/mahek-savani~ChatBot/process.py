import os
import sys
from typing import List, Tuple

import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

#OpenAI API Key
os.environ["OPENAI_API_KEY"] = "API-Key"

#global constants
DATA_DIRECTORY = "data/"

# process data from post request
class ChatInput(BaseModel):
    prompt: str

app = FastAPI()

#allowed origins for CORS
origins = [
    "http://localhost:3000",
    "localhost:3000"
]

#middlewares for handling CORS request
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# FastAPI route that accepts POST requests and returns chat responses
@app.post("/chat")
async def handle_chat(chat_input: ChatInput):
    initial_query = chat_input.prompt
    index = load_or_create_index(DATA_DIRECTORY)
    chain = setup_conversation_chain(index)
    chat_history = []
    result = chat_loop(chain, chat_history, initial_query)
    return result

# function to create index for chat-app
def load_or_create_index(directory: str) -> VectorStoreIndexWrapper:
    loader = DirectoryLoader(directory)
    return VectorstoreIndexCreator().from_loaders([loader])

#function to setup the chat conversation chain
def setup_conversation_chain(index):
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

#function to perform the chat loop, it receives a query and returns a response to the post request from front-end (react) app
def chat_loop(chain, chat_history: List[Tuple[str, str]], initial_query: str = None):
    query = initial_query
    result = chain({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    return result

#main function that runs the chat
def main():
    initial_query = sys.argv[1] if len(sys.argv) > 1 else None
    index = load_or_create_index(DATA_DIRECTORY)
    chain = setup_conversation_chain(index)
    chat_history = []
    chat_loop(chain, chat_history, initial_query)

if __name__ == "__main__":
    main()