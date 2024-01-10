import os
import openai

from llama_index.response.schema import Response, StreamingResponse
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
from llama_index.agent import OpenAIAgent
from llama_index.tools import QueryEngineTool
# agent = OpenAIAgent.from_tools(tools=[query_engine_tool], verbose=True)

STREAMING = True

try:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    index = load_index_from_storage(storage_context)
except:
    from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader("./data").load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    index.storage_context.persist()


@cl.on_chat_start
async def factory():
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            streaming=STREAMING,
        ),
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        chunk_size=512,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )

    query_engine = index.as_query_engine(
        service_context=service_context,
        streaming=STREAMING,
    )
    
    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="query_tool",
        description=(
            "Useful for questions needing context from the documents."
    ),
)


    agent = OpenAIAgent.from_tools(tools=[query_engine_tool], verbose=True)
    cl.user_session.set("agent", agent)
    cl.user_session.set("query_engine", query_engine)
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful QA assistant. you are specialised to know extra context about local files."}],
    )


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})
    response = await cl.make_async(query_engine.query)(message.content)

    response_message = cl.Message(content="")

    if isinstance(response, Response):
        response_message.content = str(response)
        await response_message.send()
    elif isinstance(response, StreamingResponse):
        gen = response.response_gen
        for token in gen:
            await response_message.stream_token(token=token)

        if response.response_txt:
            response_message.content = response.response_txt

        await response_message.send()