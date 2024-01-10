import os
import openai
import logging
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

# Set up logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

try:
    # Attempt to rebuild storage context and load index
    logger.info("Attempting to load index from storage.")
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
except Exception as e:
    # If index loading fails, create a new index
    logger.warning(f"Failed to load index from storage: {e}. Creating a new index.")
    from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader("./data").load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    index.storage_context.persist()
    logger.info("New index created and persisted.")

@cl.on_chat_start
async def factory():
    #embed_model = OpenAIEmbedding()
    chunk_size = 1000

    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0,
            model_name="gpt-4",
            streaming=True,
        ),
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        chunk_size=chunk_size,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )

    query_engine = index.as_query_engine(
        service_context=service_context,
        streaming=True,
    )
    logger.info("Query engine initialized.") # to facilitate debugging and monitoring
    cl.user_session.set("query_engine", query_engine) 

@cl.on_message
async def main(message):
    try:
        query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine
        logger.info(f"Received message: {message}")
        response = await cl.make_async(query_engine.query)(message)
        response_message = cl.Message(content="")

        # Logic to prepare answer and source_elements
        for token in response.response_gen:
            await response_message.stream_token(token=token)

        if response.response_txt:
            response_message.content = response.response_txt

        # Integrated new message object
        if answer: # conditional to when is not None
            await cl.Message(content=answer, elements=source_elements).send()
        
        await response_message.send()
        logger.info(f"Response sent: {response.response_txt}")

    except Exception as e:
        logger.error(f"An error occurred while processing the message: {e}")


