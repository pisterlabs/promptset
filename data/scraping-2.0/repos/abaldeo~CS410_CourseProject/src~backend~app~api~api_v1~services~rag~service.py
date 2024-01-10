from app.api.api_v1.services.rag.core import create_conversation_chain, create_memory, create_qa_chain_with_sources, create_redis_history, format_qa_response, get_llm
from fastapi import APIRouter
from app.api.api_v1.services.embedding.core import (
                                                    get_embedding_model, 
                                                    create_vectorstore)
from app.api.api_v1.services.embedding.utils import Timer 

# from langchain.storage.upstash_redis import UpstashRedisStore
# from upstash_redis import Redis
import functools
from langchain.storage.redis import RedisStore
import redis
from langchain.embeddings import CacheBackedEmbeddings

# from dotenv import load_dotenv
# import os 
from pydantic import BaseModel
from app.core.config import settings, get_settings
from loguru import logger
from lazy_load import lazy, lazy_func

from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
set_llm_cache(InMemoryCache())

# load_dotenv()

router = r = APIRouter()

@functools.lru_cache()
def get_redis_instance():
    # redis_client = Redis(url=REDIS_URL, token=REDIS_TOKEN, rest_retries=5, rest_retry_interval=3, allow_telemetry=False)    
    REDIS_URL = settings.EMBEDDING_REDIS_URL
    REDIS_TOKEN = settings.EMBEDDING_REDIS_TOKEN
    REDIS_HOST = settings.EMBEDDING_REDIS_HOST
    REDIS_PORT = settings.EMBEDDING_REDIS_PORT 
    REDIS_PASSWD = settings.EMBEDDING_REDIS_PASSWD
    redis_client = redis.Redis( host=REDIS_HOST, port=REDIS_PORT,password=REDIS_PASSWD)
    return redis_client

# redis_instance = get_redis_instance()
# REDIS_STORE  = RedisStore(client= redis_instance, ttl=None, namespace="rag_service")

@lazy_func
def get_redis_store():
    logger.info("Creating RedisStore")
    redis_instance = get_redis_instance()
    return RedisStore(client= redis_instance, ttl=None, namespace="rag_service")

REDIS_STORE  = get_redis_store()
GPT_MODEL_NAME = settings.GPT_MODEL_NAME
EMBEDDING_MODEL_NAME = settings.EMBEDDING_MODEL_NAME

# EMEDDING_MODEL  = get_embedding_model(settings.EMBEDDING_MODEL_NAME)
# EMBEDDING_CACHE = UpstashRedisStore(client=redis_client, ttl=None, namespace="embedding_service")



@lazy_func
def get_embedder(embedding_model_name, redis_store, gpt_model_name):
    logger.info("Creating Embedder")
    embedding_model = get_embedding_model(embedding_model_name)
    return CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embedding_model, 
    document_embedding_cache=redis_store, 
    namespace=gpt_model_name)
    
# EMBEDDER = CacheBackedEmbeddings.from_bytes_store(
#     underlying_embeddings=EMEDDING_MODEL, 
#     document_embedding_cache=REDIS_STORE, 
#     namespace=GPT_MODEL_NAME
# )
EMBEDDER = get_embedder(EMBEDDING_MODEL_NAME, REDIS_STORE, GPT_MODEL_NAME)


# VECTOR_DB =  create_vectorstore(embedding_model=EMBEDDER)
VECTOR_DB =  lazy(create_vectorstore, EMBEDDER)

LLM = lazy(get_llm)

 
 
 
def clean_query_text(text):
    import string, re 
    # Lowercase the text
    text = text.lower().strip()
 
   # Remove extra spaces/lines
    text = re.sub(r'[ |\t]+', ' ', text)
    text = re.sub(r"\n+", ' ', text)

    lines = (line.strip() for line in text.splitlines())
    text = " ".join(iter(lines))

    punc_chars = string.punctuation.replace('.','')
    # remove punctuations
    text = ''.join(c for c in text if c not in punc_chars)    
    return text


@r.get("/qa")
def qa(question: str):
    qa_chain = create_qa_chain_with_sources(LLM, VECTOR_DB)
    try:
        with Timer() as t:
            question = clean_query_text(question)            
            return format_qa_response(qa_chain({"question": question}))
    except Exception as e:
        logger.exception(e)
        return "Sorry, something went wrong..."
    

from fastapi import WebSocket, WebSocketDisconnect, Depends, Request
from app.api.api_v1.services.rag.callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from app.api.api_v1.services.rag.schemas import ChatResponse
import uuid 

BAD_ANSWER="Hmm, I am not sure" 

@r.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket, session: str = Depends(lambda: Request.state.session)):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    # chat_history = []
    # qa_chain = create_qa_chain_with_sources(LLM, VECTOR_DB)
    print(session)
    session_id  =  session.get("session_id")
    if not  session_id:
        session_id = str(uuid.uuid4())
        session.update({"session_id": session_id})
    history = create_redis_history(session_id=session_id)
    memory = create_memory(history)
    qa_chain = create_conversation_chain(LLM, VECTOR_DB, memory, callbacks=[question_handler, stream_handler])
        
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    #qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall( {"question": question, })

            if not BAD_ANSWER in result["answer"]:
                source_message = format_qa_response(result)
                source_resp = ChatResponse(sender="bot", message=source_message, type="stream")
                await websocket.send_json(source_resp.dict())
            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logger.info("websocket disconnect")
            break
        except Exception as e:
            logger.exception(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())
    