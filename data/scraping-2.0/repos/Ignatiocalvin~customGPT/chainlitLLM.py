import os
import openai
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.callbacks.base import CallbackManager
from llama_index import (
 LLMPredictor,
 ServiceContext,
 StorageContext,
 load_index_from_storage,
)
from langchain.chat_models import ChatOpenAI
import chainlit as cl


try:
  storage_context = StorageContext.from_defaults(persist_dir="./docs")
  index = load_index_from_storage(storage_context)
except:
  from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
 
  documents = SimpleDirectoryReader("./docs").load_data()
  index = GPTVectorStoreIndex.from_documents(documents)
  index.storage_context.persist()

@cl.on_chat_start
async def factory():
  llm_predictor = LLMPredictor(
     llm=ChatOpenAI(
     temperature=0,
     model_name="gpt-3.5-turbo",
     streaming=True,
     ),
   )
   
  service_context = ServiceContext.from_defaults(
   llm_predictor=llm_predictor,
   chunk_size=512,
   callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
  )
  
  query_engine = index.as_query_engine(
   service_context=service_context,
   streaming=True,
  )
  # Store query in user session
  cl.user_session.set("query_engine", query_engine)

# Define asynchronous function which takes string input. This function will be called when a 
# message is received thanks to the decorator.
@cl.on_message
async def main(message):
    # Get query from session
    query_engine = cl.user_session.get("query_engine") # type: RetrieverQueryEngine
    
    # Call query asynchronously
    response = await cl.make_async(query_engine.query)(message)
    response_message = cl.Message(content="")

    for token in response.response_gen:
        await response_message.stream_token(token=token)

    if response.response_txt:
        response_message.content=response.response_txt
   
    await response_message.send()
