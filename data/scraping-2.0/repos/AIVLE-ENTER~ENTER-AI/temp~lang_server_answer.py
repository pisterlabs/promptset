from fastapi import FastAPI
# from langchain.prompts import ChatPromptTemplate
# from langchain.chat_models import ChatOpenAI
# from langserve import add_routes

# import pandas as pd
# from langchain.document_loaders import DataFrameLoader
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.prompts import PromptTemplate
# from langchain.schema import format_document
# from langchain.memory import ConversationBufferMemory
# from langchain_core.runnables import RunnableParallel,RunnablePassthrough, RunnableLambda
# from operator import itemgetter
# from langchain_core.messages import get_buffer_string
# from langchain_core.output_parsers import StrOutputParser
# from langchain.retrievers.multi_query import MultiQueryRetriever
#from typing import Optional
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import json
import time
import pickle

from mkchain import make_chain


class Quest(BaseModel):
    question: str

class UserOut(BaseModel):
    answer: str
    
class Topic(BaseModel):
    keyword: str

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

# add_routes(
#     app,
#     make_chain('faiss_index'),
#     path="/joke",
#     #enabled_endpoints=("invoke", "batch", "stream"),
# )
# async def streamer(chain,input):
#     for s in chain.stream(input):
#         current_time = time.strftime("%Y-%m-%d %H:%M:%S").encode('utf-8')
#         yield "data: " +s['answer'].content+str(current_time) + '\n\n'
        
@app.post('/answer/{keyword}')
async def answer(keyword: str, item: Quest):
        chain, memory = make_chain(keyword,'m.txt')
        input = {'question':item.question}
        result = chain.invoke(input)
        memory.save_context(input, {"answer": result["answer"].content})
        with open('m.txt','wb') as f:
            pickle.dump(memory,f)
            
        return result
    #return StreamingResponse(streamer(chain,input),media_type="text/event-stream")

# @app.post('/crawl')
# async def answer(item: Topic):
    
#     return result

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)